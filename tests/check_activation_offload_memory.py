# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measure native saved-activation offload; separate from full Super acceptance."""
import argparse
import copy
import gc
import json
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers.modeling_outputs import CausalLMOutputWithPast

from check_activation_offload_native import load


class Stack(nn.Module):
    def __init__(self, hidden, layers):
        super().__init__()
        self.embed = nn.Embedding(1024, hidden)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, hidden, bias=False), nn.Tanh(), nn.Dropout(.1))
            for _ in range(layers)
        ])
        self.lm_head = nn.Linear(hidden, 1024, bias=False)

    def forward(self, input_ids, logits_to_keep=0, output_hidden_states=False):
        """Map integer [batch, sequence] IDs to logits [batch, sequence, vocab]."""
        hidden = self.embed(input_ids)
        for block in self.blocks:
            hidden = checkpoint(block, hidden, use_reentrant=False)
        return CausalLMOutputWithPast(logits=self.lm_head(hidden))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rl-root', type=Path, required=True)
    parser.add_argument('--policy-config', type=Path, required=True)
    args = parser.parse_args()
    overlay = args.rl_root / 'nemo_rl/models/automodel'
    load('nemo_rl.models.automodel.linear_logprobs', overlay / 'linear_logprobs.py')
    train = load('nemo_rl.models.automodel.train', overlay / 'train.py')
    from nemo_rl.models.automodel.data import ProcessedInputs, ProcessedMicrobatch
    from nemo_rl.algorithms.loss.loss_functions import ClippedPGLossFn, ClippedPGLossConfig
    from nemo_rl.algorithms.logits_sampling_utils import TrainingSamplingParams
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict

    base = json.loads(args.policy_config.read_text())
    sampling = TrainingSamplingParams(temperature=.9)
    results = []
    # CPU FP32 first, then GPU BF16 with a long enough stack to measure storage.
    for device, dtype, length, hidden, layers in [('cpu', torch.float32, 19, 32, 3),
                                                ('cuda', torch.bfloat16, 32768, 1024, 12)]:
        reports, reference = [], None
        for offload in (False, True):
            torch.manual_seed(991)
            model = Stack(hidden, layers).to(device=device, dtype=dtype)
            cfg = copy.deepcopy(base['policy'])
            cfg['dtensor_cfg']['linear_logprob_chunk_size'] = 256
            cfg['dtensor_cfg']['activation_cpu_offload'] = offload
            ids = torch.randint(1024, (1, length), device=device)
            data = BatchedDataDict({'input_ids': ids, 'input_lengths': torch.tensor([length], device=device),
                'token_mask': torch.ones_like(ids, dtype=torch.float32),
                'sample_mask': torch.ones(1, device=device),
                'advantages': torch.randn(1, length, device=device)})
            data['token_mask'][:, :3] = 0
            processed = ProcessedMicrobatch(data, ProcessedInputs(ids, length), 1, length)
            prepared = train.PreparedModelForward({'input_ids': ids}, 1, None, nullcontext)
            def run(post):
                torch.manual_seed(123)
                return train.forward_with_post_processing_fn(model, prepared, post, processed,
                    global_valid_seqs=torch.tensor(1., device=device),
                    global_valid_toks=data['token_mask'].sum(), sampling_params=sampling)
            with torch.no_grad():
                lp, _, _ = run(train.LogprobsPostProcessor(cfg, False, sampling))
            data['prev_logprobs'] = lp + .01
            data['generation_logprobs'] = lp - .03
            data['reference_policy_logprobs'] = lp
            if device == 'cuda':
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            loss, _, _ = run(train.LossPostProcessor(
                ClippedPGLossFn(ClippedPGLossConfig(**base['loss_fn'])), cfg, None, 1, 1, False, sampling))
            forward_allocated = torch.cuda.memory_allocated() if device == 'cuda' else None
            loss.backward()
            if device == 'cuda':
                torch.cuda.synchronize()
            gradients = {name: p.grad.float().cpu().clone() for name, p in model.named_parameters()}
            assert all(torch.isfinite(g).all() for g in gradients.values())
            assert sum(float(g.square().sum()) for g in gradients.values()) > 0
            if reference is None:
                reference = gradients
            else:
                for name, value in gradients.items():
                    torch.testing.assert_close(value, reference[name], atol=0, rtol=0)
            reports.append({'offload': offload, 'loss': float(loss.detach()),
                            'allocated_after_forward_bytes': forward_allocated,
                            'peak_allocated_bytes': torch.cuda.max_memory_allocated() if device == 'cuda' else None})
            del model, data, processed, prepared, lp, loss, gradients
            gc.collect()
        assert reports[0]['loss'] == reports[1]['loss']
        if device == 'cuda':
            assert reports[1]['allocated_after_forward_bytes'] < reports[0]['allocated_after_forward_bytes'] * .5, reports
            assert reports[1]['peak_allocated_bytes'] < reports[0]['peak_allocated_bytes'], reports
        results.append({'device': device, 'dtype': str(dtype), 'length': length, 'layers': layers,
                        'exact_gradient_parity': True, 'reports': reports})
    with args.output.open('x') as stream:
        json.dump({'passed': True, 'cases': results, 'full_super_update': False}, stream, indent=2)
    print(json.dumps(results), flush=True)


if __name__ == '__main__':
    main()
