# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Saved-activation offload parity through native GRPO and four-rank FSDP."""
import argparse
from contextlib import nullcontext
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard
from torch.utils.checkpoint import checkpoint
from transformers.modeling_outputs import CausalLMOutputWithPast


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(1024, 64)
        self.body = nn.Sequential(nn.Linear(64, 64, bias=False), nn.Tanh(), nn.Dropout(.1))
        self.lm_head = nn.Linear(64, 1024, bias=False)

    def forward(self, input_ids, logits_to_keep=0, output_hidden_states=False):
        hidden = self.embed(input_ids)
        hidden = checkpoint(self.body, hidden, use_reentrant=False).tanh()
        selected = hidden[:, -logits_to_keep:] if logits_to_keep else hidden
        return CausalLMOutputWithPast(logits=self.lm_head(selected),
                                     hidden_states=(hidden,) if output_hidden_states else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rl-root", type=Path, required=True)
    parser.add_argument("--policy-config", type=Path, required=True)
    args = parser.parse_args()
    overlay = args.rl_root / "nemo_rl/models/automodel"
    linear = load("nemo_rl.models.automodel.linear_logprobs", overlay / "linear_logprobs.py")
    train = load("nemo_rl.models.automodel.train", overlay / "train.py")
    from nemo_rl.models.automodel.data import ProcessedInputs, ProcessedMicrobatch
    from nemo_rl.algorithms.loss.loss_functions import ClippedPGLossFn, ClippedPGLossConfig
    from nemo_rl.algorithms.logits_sampling_utils import TrainingSamplingParams
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict

    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    try:
        assert dist.get_world_size() == 4
        mesh = init_device_mesh("cuda", (4,), mesh_dim_names=("dp",))
        base_config = json.loads(args.policy_config.read_text())
        sampling = TrainingSamplingParams(temperature=.9)
        cases = []
        for offload in (False, True):
            torch.manual_seed(19)
            reference = Model().cuda()
            candidate = copy.deepcopy(reference)
            policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                                          output_dtype=torch.float32)
            for model in (reference, candidate):
                kwargs = dict(mesh=mesh, mp_policy=policy)
                if offload:
                    kwargs["offload_policy"] = CPUOffloadPolicy(pin_memory=False)
                for child in (model.embed, model.body, model.lm_head):
                    fully_shard(child, **kwargs)
                fully_shard(model, **kwargs)
            torch.manual_seed(127 + rank)
            ids = torch.randint(1024, (2, 35), device="cuda")
            lengths = torch.tensor([35, 29], device="cuda")
            token_mask = torch.ones_like(ids, dtype=torch.float32)
            token_mask[:, ::3] = 0
            token_mask[1, 29:] = 0
            data = BatchedDataDict({"input_ids": ids, "input_lengths": lengths,
                "token_mask": token_mask, "sample_mask": torch.tensor([1., 0.], device="cuda"),
                "advantages": torch.randn(2, 35, device="cuda")})
            processed = ProcessedMicrobatch(data, ProcessedInputs(ids, 35), 2, 35)
            prepared = train.PreparedModelForward({"input_ids": ids}, 1, None, nullcontext)
            plain_cfg = copy.deepcopy(base_config["policy"])
            plain_cfg["dtensor_cfg"]["linear_logprob_chunk_size"] = 13
            plain_cfg["dtensor_cfg"]["activation_cpu_offload"] = False
            chunk_cfg = copy.deepcopy(plain_cfg)
            chunk_cfg["dtensor_cfg"]["activation_cpu_offload"] = True
            linear.validate_projection_config(chunk_cfg, sampling, "nemotron_h_omni")

            def run(model, cfg, post):
                torch.manual_seed(20260 + rank)
                return train.forward_with_post_processing_fn(
                    model=model, prepared=prepared, post_processing_fn=post,
                    processed_mb=processed, sampling_params=sampling,
                    global_valid_seqs=torch.tensor(4., device="cuda"),
                    global_valid_toks=(token_mask * data["sample_mask"][:, None]).sum() * 4)

            with torch.no_grad():
                expected, _, _ = run(reference, plain_cfg, train.LogprobsPostProcessor(plain_cfg, False, sampling))
                actual, _, _ = run(candidate, chunk_cfg, train.LogprobsPostProcessor(chunk_cfg, False, sampling))
            torch.testing.assert_close(actual, expected, atol=.003, rtol=1e-4)
            assert not actual[:, 0].any() and not actual[1, 29:].any()
            data["prev_logprobs"] = expected + .01
            data["generation_logprobs"] = expected - .03
            data["reference_policy_logprobs"] = expected
            losses, metrics = [], []
            for model, cfg in ((reference, plain_cfg), (candidate, chunk_cfg)):
                fn = ClippedPGLossFn(ClippedPGLossConfig(**base_config["loss_fn"]))
                post = train.LossPostProcessor(fn, cfg, None, 1, 4, False, sampling)
                loss, met, _ = run(model, cfg, post)
                losses.append(float(loss.detach()))
                loss.backward()
                metrics.append(met)
            assert abs(losses[0] - losses[1]) < .0001, losses
            gradients = {}
            for (name, ref), (_, candidate_param) in zip(reference.named_parameters(), candidate.named_parameters(), strict=True):
                assert ref.grad is not None and candidate_param.grad is not None, name
                rg, cg = ref.grad.to_local(), candidate_param.grad.to_local()
                relative = float((cg-rg).norm() / rg.norm().clamp_min(1e-12))
                cosine = float(torch.nn.functional.cosine_similarity(cg.flatten(), rg.flatten(), dim=0))
                assert relative < 1e-6 and cosine > .999999, (offload, name, relative, cosine)
                gradients[name] = {"relative_l2": relative, "cosine": cosine}
            for model in (reference, candidate):
                torch.optim.AdamW(model.parameters(), lr=5e-7).step()
            for (name, ref), (_, candidate_param) in zip(reference.named_parameters(), candidate.named_parameters(), strict=True):
                delta = float((ref.to_local().detach() - candidate_param.to_local().detach()).abs().max())
                assert delta < 1.1e-6, (name, delta)
                gradients[name]["adamw_parameter_max_abs"] = delta
            assert "forward" not in candidate.lm_head.__dict__
            # Failure must restore the original head, not leave the next request patched.
            original_forward = candidate.forward
            def broken(**kwargs):
                raise ValueError("synthetic pre-head failure")
            candidate.forward = broken
            try:
                linear.model_next_token_logprobs(candidate, {"input_ids": ids}, ids,
                                                chunk_size=13, sampling_params=sampling)
            except ValueError as exc:
                assert str(exc) == "synthetic pre-head failure"
            else:
                raise AssertionError("Synthetic failure did not propagate")
            finally:
                candidate.forward = original_forward
            assert "forward" not in candidate.lm_head.__dict__
            cases.append({"cpu_offload": offload, "activation_checkpointing": True, "candidate_activation_cpu_offload": True,
                          "logprob_max_abs": float((actual-expected).abs().max()),
                          "native_grpo_losses": losses, "gradients": gradients,
                          "failed_forward_restores_head": True})
            del reference, candidate
            dist.barrier()
        # Reject a flag that would silently do nothing on an unsupported path.
        rejected = []
        for field, value in (("linear_logprob_chunk_size", None),
                             ("activation_checkpointing", False),
                             ("activation_cpu_offload", "true")):
            invalid = copy.deepcopy(chunk_cfg)
            invalid["dtensor_cfg"][field] = value
            try:
                linear.validate_projection_config(invalid, sampling, "nemotron_h_omni")
            except ValueError:
                rejected.append(field)
            else:
                raise AssertionError("Unsupported activation config accepted: " + field)
        args.output.mkdir(parents=True, exist_ok=True)
        with (args.output / f"rank{rank}.json").open("x") as stream:
            json.dump({"passed": True, "rank": rank, "cases": cases,
                       "rejected_configs": rejected, "full_super_update": False}, stream, indent=2)
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
