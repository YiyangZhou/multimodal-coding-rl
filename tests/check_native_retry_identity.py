"""Fault injection through the pinned native capture/recovery path; CPU only.

The stream transport and replay storage are test doubles. Retry dispatch,
completion callbacks, recovery identity and finalization use native RL code.
"""
import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from mmrl.config import training_config, validate_capture_retries
from nemo_rl.algorithms.async_utils.replay_buffer import DataPlaneCheckpointBarrier
from nemo_rl.algorithms.single_controller_utils.config import RolloutRecoveryConfig
from nemo_rl.experience import rollout_manager as native
from nemo_rl.experience.failures import GymTransportError
from nemo_rl.experience.interfaces import Completion
from nemo_rl.experience.rollout_recovery import RolloutRecoveryLedger
from nemo_rl.utils.timer import Timer


class Buffer:
    def __init__(self):
        self.data_plane_checkpoint_barrier = DataPlaneCheckpointBarrier()
        self.reservations = []
        self.aborts = []

    def reserve(self, **kwargs):
        self.reservations.append(kwargs)
        return kwargs["group_id"]

    def abort(self, group_id):
        self.aborts.append(group_id)


def completion(rollout_id):
    return Completion(
        message_log=[], truncated=False, reward=0.7,
        env_extras={
            "ng_rollout_id": rollout_id,
            "ng_receipt": {
                "rollout_id": rollout_id,
                "manifest": [{"staging_key": rollout_id + "/call"}],
            },
        },
    )


async def exercise(max_row_attempts):
    buffer = Buffer()
    manager = object.__new__(native.RolloutManager)
    manager._tokenizer = None
    manager._num_generations_per_prompt = 8
    manager._rollout_recovery_config = RolloutRecoveryConfig()
    manager._tq_buffer = buffer
    manager._weight_version = 7
    manager._retry_policy = native.RolloutRetryPolicy(
        max_infra_attempts=5, max_data_attempts=1,
        max_gym_row_attempts=max_row_attempts, backoff_base_s=0, max_backoff_s=0,
    )
    manager._stats = native.RolloutStats()
    manager._consecutive_infra_drops = 0
    manager._recovery_ledger = RolloutRecoveryLedger()
    manager._data_plane_checkpoint_barrier = buffer.data_plane_checkpoint_barrier
    worker = object.__new__(native.AsyncNemoGymRolloutImpl)
    worker._task_to_env = {"nemo_gym": object()}
    worker._num_generations_per_prompt = 8
    worker._max_gym_row_attempts = max_row_attempts
    worker._timeouts = SimpleNamespace(rollout_s=10)
    worker._deadline_registry = None
    worker._stats = manager._stats
    calls, streamed, callbacks = [], [], []

    async def stream(_env, pending, *args, on_completion):
        streamed.append([row["_ng_rollout_id"] for row in pending])
        # One sibling finished before the transport failed. Seven remote
        # commands may still be alive; their attempts must not be reused.
        row = pending[0]
        await on_completion(row["_rowidx"], completion(row["_ng_rollout_id"]))
        raise GymTransportError("injected lost stream after partial completion")

    worker._stream_rows = stream

    class Transport:
        async def run_rollout(self, sample, *, rollout_ids, generation_indices,
                              on_completion, recovery_granularity):
            calls.append((list(rollout_ids), list(generation_indices)))
            callbacks.append(on_completion)
            if len(calls) == 1:
                inputs = [{"_rowidx": i, "_ng_rollout_id": rollout_ids[i]}
                          for i in generation_indices]
                await worker._run_rollouts(
                    inputs, Timer(), "fault_probe", on_completion=on_completion,
                    recovery_granularity=recovery_granularity,
                )
            else:
                for i in generation_indices:
                    await on_completion(i, completion(rollout_ids[i]))

    manager._impl = Transport()
    shard = SimpleNamespace(pick_handle=lambda _: object(), instance_label=lambda _: "probe")
    with patch.object(native, "as_nemo_gym_shard_set", return_value=shard), \
         patch.object(native, "get_nemo_gym_route_name", return_value="mm_opencode"):
        request = await manager.generate_for_finalization({"idx": 0, "extra_env_info": {}})
    assert len(streamed) == 1, "Inner retry reused physical capture IDs"
    assert len(calls) == 2
    first, second = calls[0][0], calls[1][0]
    assert calls[0][1] == list(range(8))
    assert calls[1][1] == list(range(1, 8)), "Completed sibling was regenerated"
    assert second[0] == first[0]
    assert set(first[1:]).isdisjoint(second[1:])
    assert request.rollout_ids == tuple(second)
    assert [r["rollout_id"] for r in request.receipts] == second
    assert len(set(request.canonical_sample_ids)) == 8
    # A late result from the abandoned physical attempt cannot replace the
    # successful retry, even when it arrives through the original callback.
    try:
        await callbacks[0](1, completion(first[1]))
    except ValueError:
        pass
    else:
        raise AssertionError("Abandoned attempt accepted a late completion")
    final = manager._recovery_ledger.finalization_inputs(request.group_id)
    assert tuple(final[0]) == request.rollout_ids
    assert len(buffer.aborts) == 1
    return {"physical_dispatches": 2, "retained_siblings": 1, "fresh_retries": 7,
            "late_abandoned_completion_rejected": True}


if __name__ == "__main__":
    base = json.loads(Path(os.environ["MMRL_RETRY_BASE"]).read_text())
    cfg = training_config(base, "/probe/gym.json", "/probe/data.jsonl", "/probe/run")
    validate_capture_retries(cfg)
    for invalid in (None, True, 3):
        bad = {"async_rl": {"rollout_failure": {"nemo_gym": {"max_row_attempts": invalid}}}}
        try:
            validate_capture_retries(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("Unsafe retry configuration accepted")
    attempts = cfg["async_rl"]["rollout_failure"]["nemo_gym"]["max_row_attempts"]
    result = asyncio.run(exercise(attempts))
    # Confirm this test catches the original default rather than only testing
    # a happy path with freshly minted synthetic IDs.
    try:
        asyncio.run(exercise(3))
    except AssertionError as error:
        assert str(error) == "Inner retry reused physical capture IDs"
    else:
        raise AssertionError("Original retry behavior unexpectedly passed")
    print(json.dumps(dict(passed=True, original_behavior_rejected=True, **result)), flush=True)
