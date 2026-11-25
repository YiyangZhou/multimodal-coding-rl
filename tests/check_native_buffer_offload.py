"""Run with torchrun on GPUs: native setup, DTensor buffers and DCP roundtrip.

The model factory supplies a tiny module instead of downloading a checkpoint.
The actual setup function, device transfers, optimizer and DCP APIs are exercised.
This does not establish full Super checkpoint compatibility.
"""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    StateDictOptions,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor
from nemo_automodel.components.distributed.config import FSDP2Config
from nemo_rl.models.automodel.config import DistributedContext, RuntimeConfig
from nemo_rl.models.automodel.setup import setup_model_and_optimizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="cuda:nccl,cpu:gloo")
    mesh = init_device_mesh(
        "cuda", (dist.get_world_size(),), mesh_dim_names=("dp_shard",)
    )
    args.output.mkdir(parents=True, exist_ok=True)

    def local(tensor):
        return tensor.to_local() if isinstance(tensor, DTensor) else tensor

    results = []
    try:
        for name, placement in [
            ("plain", None),
            ("replicate", Replicate()),
            ("shard", Shard(0)),
        ]:

            def make_model(*, empty):
                module = torch.nn.Module()
                module.config = SimpleNamespace(
                    pad_token_id=0, tie_word_embeddings=False
                )
                module.register_parameter(
                    "weight", torch.nn.Parameter(torch.ones(2, device="cuda"))
                )
                buf = torch.arange(16, dtype=torch.float32, device="cuda")
                if placement is not None:
                    buf = distribute_tensor(buf, mesh, [placement])
                module.register_buffer(
                    "persistent", torch.zeros_like(buf) if empty else buf
                )
                module.register_buffer("ephemeral", buf.clone(), persistent=False)
                return module

            source = make_model(empty=False)
            expected = {
                key: local(value).cpu().clone() for key, value in source.named_buffers()
            }
            checkpoint = args.output / name
            dcp.save(
                {
                    "model": get_model_state_dict(
                        source, options=StateDictOptions(cpu_offload=True)
                    )
                },
                checkpoint_id=checkpoint,
            )

            class Factory:
                @staticmethod
                def from_pretrained(*args, **kwargs):
                    return make_model(empty=True)

            class CheckpointReader:
                def load_checkpoint(
                    self, *, model, weights_path, optimizer, optimizer_path, scheduler
                ):
                    assert isinstance(optimizer, torch.optim.AdamW)
                    assert optimizer_path is None
                    state = get_model_state_dict(
                        model, options=StateDictOptions(cpu_offload=True)
                    )
                    assert set(state) == {"weight", "persistent"}
                    dcp.load({"model": state}, checkpoint_id=weights_path)

            config = {
                "model_name": "tiny-buffer-regression",
                "optimizer": {"name": "torch.optim.AdamW", "kwargs": {"lr": 1e-4}},
                "dtensor_cfg": {
                    "expert_parallel_size": 1,
                    "activation_checkpointing": False,
                },
            }
            runtime = RuntimeConfig(
                model_class=Factory,
                model_config=SimpleNamespace(
                    architectures=["GPT2LMHeadModel"], model_type="gpt2"
                ),
                hf_config_overrides={},
                allow_flash_attn_args=True,
                attn_impl=None,
                dtype=torch.float32,
                enable_seq_packing=False,
                max_grad_norm=1.0,
                cpu_offload=True,
                offload_optimizer_for_logprob=False,
                is_generation_colocated=False,
                sampling_params=None,
                is_reward_model=False,
            )
            context = DistributedContext(
                mesh, None, FSDP2Config(), None, dist.get_world_size(), 1, 1
            )
            state = setup_model_and_optimizer(
                config=config,
                tokenizer=SimpleNamespace(pad_token_id=0),
                runtime_config=runtime,
                distributed_context=context,
                checkpoint_manager=CheckpointReader(),
                weights_path=str(checkpoint),
            )
            for key, value in state.model.named_buffers():
                assert value.device.type == "cpu" and local(value).device.type == "cpu"
                torch.testing.assert_close(local(value), expected[key])
            state.model.weight.sum().backward()
            state.optimizer.step()
            assert bool((state.model.weight < 1).all())
            state.model.to("cuda")
            for key, value in state.model.named_buffers():
                assert value.device.type == "cuda"
                torch.testing.assert_close(local(value).cpu(), expected[key])
            results.append({"case": name, "native_setup_and_dcp_restore": True})
        dist.barrier()
        (args.output / f"rank{rank}.json").write_text(
            json.dumps(results, indent=2) + "\n"
        )
        print(json.dumps({"rank": rank, "results": results}), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
