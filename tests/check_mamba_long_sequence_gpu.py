"""Exercise Super's fused Mamba geometry without loading model weights."""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback

p = argparse.ArgumentParser()
p.add_argument('--length', type=int, required=True)
p.add_argument('--baseline', type=Path, required=True, help='Unpatched pinned mamba_ssm/ops/triton snapshot')
p.add_argument('--compare-baseline', action='store_true')
p.add_argument('--repeat-reference', action='store_true')
p.add_argument('--output', required=True)
args = p.parse_args()
assert not args.compare_baseline or (args.length <= 8192)
assert not args.repeat_reference or (args.length % 8192 == 0 and not args.compare_baseline)
result = {'length': args.length, 'installed_patch': True, 'stage': 'imports',
          'scope': 'Synthetic fused Mamba forward/backward at Super dimensions; not full-model or multimodal acceptance.'}
start = time.monotonic()

def checkpoint(stage):
    result['stage'] = stage
    result['elapsed_seconds'] = time.monotonic() - start
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')

try:
    import torch
    from mamba_ssm.ops.triton import ssd_chunk_state, ssd_combined, layernorm_gated
    from mamba_baseline import verify_and_reference
    restore_kernels, hashes = verify_and_reference(args.baseline)
    result['installed_source_sha256'] = hashes
    # Record the first nonfinite stage rather than reporting only the final output.
    def trace_operation(name, function):
        def checked(*arguments, **keywords):
            values = function(*arguments, **keywords)
            for value in values if isinstance(values, (tuple, list)) else (values,):
                if isinstance(value, torch.Tensor) and value.is_floating_point():
                    if not bool(torch.isfinite(value).all()):
                        raise RuntimeError('Nonfinite tensor from ' + name + ':shape=' + str(tuple(value.shape)))
            return values
        return checked
    for name in ['causal_conv1d_fwd_function', '_chunk_cumsum_fwd', '_chunk_state_fwd',
                 '_state_passing_fwd', '_bmm_chunk_fwd', '_chunk_scan_fwd', '_layer_norm_fwd']:
        setattr(ssd_combined, name, trace_operation(name, getattr(ssd_combined, name)))
    torch.manual_seed(704)
    checkpoint('allocate')
    def parameter(shape, scale=.02, dtype=torch.bfloat16):
        return torch.empty(shape, device='cuda', dtype=dtype).normal_(0, scale).requires_grad_()
    z = parameter((1, args.length, 18560), .05)
    if args.repeat_reference:
        with torch.no_grad():
            for offset in range(8192, args.length, 8192):
                z[:, offset:offset + 8192].copy_(z[:, :8192])
    conv = parameter((10240, 4))
    conv_bias = parameter((10240,))
    bias = torch.full((128,), -4., device='cuda', requires_grad=True)
    A = (-torch.linspace(.1, 1., 128, device='cuda')).requires_grad_()
    D = torch.ones((128,), device='cuda', requires_grad=True)
    norm = torch.ones((8192,), device='cuda', dtype=torch.bfloat16, requires_grad=True)
    proj = parameter((4096, 8192), .01)
    seq = torch.zeros((1, args.length), device='cuda', dtype=torch.int32)
    seq[:, args.length // 2:] = 1
    if args.repeat_reference:
        seq.copy_((torch.arange(args.length, device='cuda') // 8192).view(1, -1))
    torch.cuda.synchronize()
    checkpoint('forward')
    out = ssd_combined.mamba_split_conv1d_scan_combined(
        z, conv, conv_bias, bias, A, D=D, chunk_size=128, seq_idx=seq,
        activation='silu', rmsnorm_weight=norm, rmsnorm_eps=1e-5,
        outproj_weight=proj, outproj_bias=None, headdim=64, ngroups=8,
        norm_before_gate=False, return_final_states=False,
    )
    torch.cuda.synchronize()
    assert out.shape == (1, args.length, 4096)
    assert bool(torch.isfinite(out).all())
    result['output_rms'] = out.float().square().mean().sqrt().item()
    checkpoint('backward')
    out.float().square().mean().backward()
    torch.cuda.synchronize()
    result['gradient_norms'] = {}
    for name, value in [('projection_input', z), ('conv', conv), ('conv_bias', conv_bias),
                        ('dt_bias', bias), ('A', A), ('D', D), ('norm', norm), ('out_proj', proj)]:
        assert value.grad is not None and bool(torch.isfinite(value.grad).all()), name
        magnitude = value.grad.float().norm().item()
        assert magnitude > 0, name
        result['gradient_norms'][name] = magnitude
    result['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    if args.repeat_reference:
        checkpoint('packed_short_reference')
        repeats = args.length // 8192
        originals = [z, conv, conv_bias, bias, A, D, norm, proj]
        reference = [z[:, :8192].detach().clone().requires_grad_()] + [
            v.detach().clone().requires_grad_() for v in originals[1:]]
        rz, rc, rcb, rb, ra, rd, rn, rp = reference
        restore_kernels()
        reference_out = ssd_combined.mamba_split_conv1d_scan_combined(
            rz, rc, rcb, rb, ra, D=rd, chunk_size=128,
            seq_idx=torch.zeros((1, 8192), device='cuda', dtype=torch.int32),
            activation='silu', rmsnorm_weight=rn, rmsnorm_eps=1e-5,
            outproj_weight=rp, outproj_bias=None, headdim=64, ngroups=8,
            norm_before_gate=False, return_final_states=False,
        )
        reference_out.float().square().mean().backward()
        torch.cuda.synchronize()
        max_error = 0.
        for offset in range(0, args.length, 8192):
            piece = out[:, offset:offset + 8192]
            torch.testing.assert_close(piece, reference_out, rtol=.01, atol=.0002)
            max_error = max(max_error, (piece - reference_out).abs().max().item())
            torch.testing.assert_close(z.grad[:, offset:offset + 8192].float() * repeats,
                                       rz.grad.float(), rtol=.04, atol=rz.grad.abs().max().item() * .002 + 1e-12)
        for value, ref_value in zip(originals[1:], reference[1:]):
            torch.testing.assert_close(value.grad, ref_value.grad, rtol=.04, atol=ref_value.grad.abs().max().item() * .002 + 1e-12)
        result['packed_short_reference'] = {'independent_sequences': repeats, 'length_per_sequence': 8192,
            'all_output_and_input_gradient_blocks_compared': True, 'parameter_gradients_compared': 7,
            'output_max_abs_error': max_error}
    if args.compare_baseline:
        checkpoint('compare_baseline')
        parameters = [z, conv, conv_bias, bias, A, D, norm, proj]
        candidate_gradients = [v.grad.clone() for v in parameters]
        for value in parameters:
            value.grad = None
        restore_kernels()
        baseline_out = ssd_combined.mamba_split_conv1d_scan_combined(
            z, conv, conv_bias, bias, A, D=D, chunk_size=128, seq_idx=seq,
            activation='silu', rmsnorm_weight=norm, rmsnorm_eps=1e-5,
            outproj_weight=proj, outproj_bias=None, headdim=64, ngroups=8,
            norm_before_gate=False, return_final_states=False,
        )
        torch.testing.assert_close(out, baseline_out, rtol=0, atol=0)
        baseline_out.float().square().mean().backward()
        torch.cuda.synchronize()
        for value, candidate_gradient in zip(parameters, candidate_gradients):
            torch.testing.assert_close(value.grad, candidate_gradient, rtol=.02, atol=candidate_gradient.abs().max().item() * .001 + 1e-12)
        result['baseline_comparison'] = {'exact_forward': True, 'gradient_tensors_compared': len(parameters)}
    result['passed'] = True
    checkpoint('complete')
except Exception as error:
    result.update(passed=False, error_type=type(error).__name__, error=str(error)[:1800], traceback=traceback.format_exc()[-7000:])
    checkpoint(result['stage'])
print(json.dumps({k: v for k, v in result.items() if k != 'traceback'}), flush=True)
sys.exit(0 if result.get('passed') else 1)
