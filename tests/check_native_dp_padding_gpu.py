"""Four-GPU native Automodel update with two real rows and two masked copies.

Compares a real FSDP update to the same unpadded policy-gradient update. Tiny
text model only: full Super, image-feedback and checkpoint acceptance remain
separate. Run with torchrun --nproc_per_node=4.
"""

import argparse
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor

from nemo_rl.algorithms.loss.loss_functions import ClippedPGLossConfig, ClippedPGLossFn
from nemo_rl.data_plane.preshard import shard_meta_for_dp
from nemo_rl.models.policy.workers.dtensor_policy_worker_v2 import (
    DTensorPolicyWorkerV2Impl,
)
from nemo_rl.utils.timer import Timer
from check_native_dp_padding import fixture, Worker


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(32, 8)
        self.head = torch.nn.Linear(8, 32, bias=False)

    def forward(self, input_ids, **kwargs):
        return self.head(self.embed(input_ids))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="cuda:nccl,cpu:gloo")
    assert dist.get_world_size() == 4
    mesh = init_device_mesh("cuda", (4,), mesh_dim_names=("dp_shard",))
    torch.manual_seed(912)
    oracle = TinyPolicy().cuda()
    initial = copy.deepcopy(oracle.state_dict())
    model = TinyPolicy().cuda()
    model.load_state_dict(initial)
    fully_shard(model, mesh=mesh, reshard_after_forward=True)
    loss_fn = ClippedPGLossFn(ClippedPGLossConfig(reference_policy_kl_penalty=0.01))
    source, meta, client = fixture(2)
    # Keep ratios close to one and preserve a nonzero update.
    with torch.no_grad():
        logits = oracle(source["input_ids"].cuda())
        previous = (
            logits[:, :-1]
            .log_softmax(-1)
            .gather(-1, source["input_ids"][:, 1:].cuda().unsqueeze(-1))
            .squeeze(-1)
            .cpu()
        )
    for key in ("prev_logprobs", "generation_logprobs", "reference_policy_logprobs"):
        source[key][:, 1:] = previous
    from nemo_rl.data_plane.column_io import write_columns

    write_columns(
        client,
        meta,
        {
            key: source[key]
            for key in (
                "prev_logprobs",
                "generation_logprobs",
                "reference_policy_logprobs",
            )
        },
    )
    shards, _ = shard_meta_for_dp(meta, dp_world=4, pad_to_dp=True)
    data = Worker(client)._fetch(shards[rank])

    # Full unpadded reference update, with identical clipping and SGD.
    oracle_optimizer = torch.optim.SGD(oracle.parameters(), lr=0.05)
    logits = oracle(source["input_ids"].cuda())
    lp = (
        logits[:, :-1]
        .log_softmax(-1)
        .gather(-1, source["input_ids"][:, 1:].cuda().unsqueeze(-1))
        .squeeze(-1)
    )
    oracle_loss, _ = loss_fn(
        next_token_logprobs=lp,
        data=source.to("cuda"),
        global_valid_seqs=torch.tensor(2.0, device="cuda"),
        global_valid_toks=torch.tensor(8.0, device="cuda"),
    )
    oracle_loss.backward()
    oracle_norm = torch.nn.utils.clip_grad_norm_(oracle.parameters(), 1.0)
    oracle_optimizer.step()

    worker = object.__new__(DTensorPolicyWorkerV2Impl)
    worker.model = model
    worker.optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    worker.scheduler = torch.optim.lr_scheduler.LambdaLR(
        worker.optimizer, lambda _: 1.0
    )
    worker.tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=0)
    worker.cfg = {
        "train_global_batch_size": 2,
        "train_micro_batch_size": 1,
        "dynamic_batching": {"enabled": False},
        "sequence_packing": {"enabled": False},
        "dtensor_cfg": {"sequence_parallel": False},
    }
    worker.dp_size = 4
    worker.dp_mesh = mesh
    worker.device_mesh = mesh
    worker.cp_size = 1
    worker.cp_mesh = None
    worker.moe_mesh = None
    worker.dtype = torch.float32
    worker.max_grad_norm = 1.0
    worker.enable_seq_packing = False
    worker.sampling_params = None
    worker.autocast_enabled = False
    worker._is_reward_model = False
    worker.allow_flash_attn_args = False
    worker._buffered_train_step = None
    worker.timer = Timer()
    worker.begin_train_step(loss_fn, gbs=2, mbs=1)
    worker.train_microbatch(data)
    metrics = worker.finish_train_step()
    assert worker._buffered_train_step is None
    assert worker.scheduler.last_epoch == 1
    assert float(metrics["grad_norm"]) > 0
    torch.testing.assert_close(
        metrics["grad_norm"].cpu(), oracle_norm.cpu(), atol=2e-6, rtol=2e-5
    )
    maximum_error = 0.0
    changed = False
    for name, tensor in model.state_dict().items():
        full = tensor.full_tensor() if isinstance(tensor, DTensor) else tensor
        expected = oracle.state_dict()[name]
        torch.testing.assert_close(full, expected, atol=2e-6, rtol=2e-5)
        maximum_error = max(maximum_error, float((full - expected).abs().max()))
        changed |= not torch.equal(full, initial[name])
    assert changed
    report = {
        "rank": rank,
        "world_size": 4,
        "real_trajectories": 2,
        "physical_rows": 4,
        "native_worker_update": True,
        "unpadded_reference_update_matches": True,
        "max_parameter_error": maximum_error,
        "scheduler_steps": 1,
        "full_super_verified": False,
        "image_feedback_verified": False,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"rank{rank}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
