"""Compare native refit metadata with real distributed weight exports.

Run with torchrun --nproc_per_node=4. Tiny tensors exercise the actual Omni
adapter, global DTensor shapes and dtype rules; no full Super refit claim.
"""

import argparse
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.nemotron_omni.state_dict_adapter import (
    NemotronOmniStateDictAdapter,
)
from nemo_automodel.components.moe.config import MoEConfig
from nemo_rl.models.policy.workers.dtensor_policy_worker_v2 import (
    DTensorPolicyWorkerV2Impl,
    dtensor_params_generator,
)


class StateModel(torch.nn.Module):
    def __init__(self, state, adapter):
        super().__init__()
        self.state = state
        self.state_dict_adapter = adapter

    def state_dict(self):
        return self.state


def adapter_args():
    moe = MoEConfig(
        n_routed_experts=4,
        n_shared_experts=1,
        n_activated_experts=2,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0.0,
        aux_loss_coeff=0.0,
        score_func="sigmoid",
        route_scale=1.0,
        dim=8,
        inter_dim=6,
        moe_inter_dim=6,
        norm_topk_prob=True,
        expert_activation="relu2",
    )
    return SimpleNamespace(), SimpleNamespace(), moe, BackendConfig(), torch.bfloat16


class ValueAdapter:
    def convert_single_tensor_to_hf(self, name, tensor, **kwargs):
        assert not tensor.is_meta
        suffix = ".positive" if float(tensor.sum()) > 0 else ".nonpositive"
        return [(name + suffix, tensor)]


class OmniSubclass(NemotronOmniStateDictAdapter):
    def convert_single_tensor_to_hf(self, name, tensor, **kwargs):
        assert not tensor.is_meta, "Subclasses must retain the value-bearing fallback"
        return super().convert_single_tensor_to_hf(name, tensor, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    mesh = init_device_mesh("cuda", (dist.get_world_size(),))
    shapes = {
        "language_model.model.layers.1.mixer.experts.gate_and_up_projs": (4, 8, 6),
        "language_model.model.layers.1.mixer.experts.down_projs": (4, 6, 8),
        "language_model.model.layers.0.mixer._fp32_params.A_log": (8,),
        "language_model.model.layers.0.mixer._fp32_params.dt_bias": (8,),
        "language_model.model.layers.1.mixer.gate.e_score_correction_bias": (4,),
        "language_model.model.embed_tokens.weight": (16, 8),
        "language_model.model.norm.weight": (8,),
        "language_model.mtp.layers.0.mixer.experts.down_projs": (4, 6, 8),
        "vision_projector.linear1.weight": (8, 6),
        "vision_projector.vision_final_layernorm.weight": (8,),
        "vision_model.radio_model.input_conditioner.norm_mean": (3,),
        "sound_encoder.layers.0.weight": (8, 6),
        "sound_projection.norm.weight": (8,),
        "language_model.model.layers.0.proj.lora_A.weight": (4, 8),
        "language_model.model.layers.0.proj.lora_B.weight": (8, 4),
    }
    comparisons = 0
    durations = []
    for dtype in (torch.float32, torch.bfloat16):
        for axis in (None, 0, 1):
            state = {}
            for name, shape in shapes.items():
                placement = (
                    Replicate() if axis is None else Shard(min(axis, len(shape) - 1))
                )
                source = torch.ones(shape, dtype=dtype, device="cuda")
                state[name] = distribute_tensor(source, mesh, [placement])
            model = StateModel(state, NemotronOmniStateDictAdapter(*adapter_args()))
            worker = object.__new__(DTensorPolicyWorkerV2Impl)
            worker.model, worker.dtype = model, torch.bfloat16
            # Real exporter gathers and adapts values, including uneven shards.
            expected = {
                name: (value.shape, value.dtype)
                for name, value in dtensor_params_generator(model, worker.dtype)
            }
            local_before = {
                name: value.to_local().clone() for name, value in state.items()
            }
            started = time.monotonic()
            with patch.object(
                DTensor,
                "full_tensor",
                side_effect=AssertionError("Metadata gathered tensor values"),
            ):
                actual = worker.prepare_refit_info()
            durations.append(time.monotonic() - started)
            assert actual == expected
            assert all(
                not name.endswith((".lora_A.weight", ".lora_B.weight"))
                for name in actual
            )
            for name, value in state.items():
                torch.testing.assert_close(
                    value.to_local(), local_before[name], rtol=0, atol=0
                )
            comparisons += len(expected)
    fallback_state = {
        "vision_projector.linear1.weight": distribute_tensor(
            torch.ones(8, 6, device="cuda"), mesh, [Shard(0)]
        )
    }
    for adapter in (None, ValueAdapter(), OmniSubclass(*adapter_args())):
        worker.model = StateModel(fallback_state, adapter)
        expected = {
            name: (value.shape, value.dtype)
            for name, value in dtensor_params_generator(worker.model, worker.dtype)
        }
        assert worker.prepare_refit_info() == expected
    report = {
        "rank": rank,
        "world_size": dist.get_world_size(),
        "passed": True,
        "export_metadata_comparisons": comparisons,
        "real_export_matches": True,
        "no_metadata_all_gather": True,
        "unknown_and_subclass_adapters_use_values": True,
        "max_metadata_seconds": max(durations),
        "full_super_refit_verified": False,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"rank{rank}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
