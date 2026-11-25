"""Native DP padding, canonical storage and policy-gradient regression on CPU.

Runs real sharding/fetch/write/loss code with the native in-memory data-plane
adapter. This is not a distributed Super update or GPU parity claim.
"""

import unittest
from types import SimpleNamespace

import torch
from nemo_rl.algorithms.loss.loss_functions import ClippedPGLossConfig, ClippedPGLossFn
from nemo_rl.data_plane.adapters.noop import NoOpDataPlaneClient
from nemo_rl.data_plane.column_io import kv_first_write
from nemo_rl.data_plane.preshard import shard_meta_for_dp
from nemo_rl.data_plane.schema import DP_PADDING_TAG
from nemo_rl.data_plane.worker_mixin import TQWorkerMixin
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


class Worker(TQWorkerMixin):
    def __init__(self, client):
        self._dp_client = client
        self.tokenizer = SimpleNamespace(pad_token_id=0)

    def _get_replica_group(self):
        return None

    def _local_coords(self):
        return {"tensor_parallel": 0, "context_parallel": 0, "pipeline_parallel": 0}


def fixture(n):
    data = BatchedDataDict(
        {
            "input_ids": torch.arange(n * 5).reshape(n, 5) + 1,
            "input_lengths": torch.full((n,), 5),
            "token_mask": torch.tensor([[0, 1, 1, 1, 1]] * n),
            "sample_mask": torch.ones(n),
            "generation_logprobs": torch.full((n, 5), -0.7),
            "prev_logprobs": torch.full((n, 5), -0.7),
            "reference_policy_logprobs": torch.full((n, 5), -0.8),
            "advantages": torch.sin(torch.arange(n * 5).reshape(n, 5).float()),
        }
    )
    client = NoOpDataPlaneClient()
    client.register_partition(
        partition_id="train", fields=list(data), num_samples=n, consumer_tasks=["train"]
    )
    meta = kv_first_write(
        data,
        sample_ids=[f"row{i}" for i in range(n)],
        dp_client=client,
        partition_id="train",
    )
    return data, meta, client


class PaddingTests(unittest.TestCase):
    def test_real_32_on_64_fetch_and_single_writer(self):
        source, meta, client = fixture(32)
        shards, inverse = shard_meta_for_dp(meta, dp_world=64, pad_to_dp=True)
        self.assertEqual(len(shards), 64)
        self.assertEqual(inverse, list(range(32)))
        self.assertIsNone(meta.tags)
        seen = []
        valid_count = 0
        for shard in shards:
            self.assertEqual(len(shard.sample_ids), 1)
            worker = Worker(client)
            data = worker._fetch(shard)
            padded = shard.tags[0][DP_PADDING_TAG]
            valid_count += int(data["sample_mask"].sum())
            if not padded:
                seen.extend(shard.sample_ids)
            else:
                self.assertEqual(int(data["token_mask"].sum()), 0)
            worker._write_back(
                shard, {"prev_logprobs": torch.full((1, 5), -999.0 if padded else 0.25)}
            )
        self.assertEqual(seen, meta.sample_ids)
        self.assertEqual(valid_count, 32)
        restored = client.get_samples(
            sample_ids=meta.sample_ids,
            partition_id="train",
            select_fields=["prev_logprobs", "sample_mask"],
        )
        torch.testing.assert_close(restored["prev_logprobs"], torch.full((32, 5), 0.25))
        torch.testing.assert_close(restored["sample_mask"], torch.ones(32))
        torch.testing.assert_close(source["sample_mask"], torch.ones(32))

    def test_multiple_logical_batches_and_partial_rank(self):
        _, meta, _ = fixture(6)
        shards, inverse = shard_meta_for_dp(
            meta, dp_world=4, batch_size=3, pad_to_dp=True
        )
        flat = [
            (key, tag[DP_PADDING_TAG])
            for s in shards
            for key, tag in zip(s.sample_ids, s.tags)
        ]
        self.assertEqual([flat[i][0] for i in inverse], meta.sample_ids)
        self.assertEqual(sum(not flag for _, flag in flat), 6)
        self.assertTrue(all(len(s.sample_ids) == 2 for s in shards))

    def test_default_still_rejects_unaligned_dispatch(self):
        _, meta, _ = fixture(3)
        with self.assertRaisesRegex(AssertionError, "multiple of shards"):
            shard_meta_for_dp(meta, dp_world=4)

    def test_divisible_batch_unchanged(self):
        _, meta, _ = fixture(8)
        shards, inverse = shard_meta_for_dp(meta, dp_world=4, pad_to_dp=True)
        self.assertIsNone(inverse)
        self.assertEqual([k for s in shards for k in s.sample_ids], meta.sample_ids)
        self.assertTrue(all(s.tags is None for s in shards))

    def test_packing_cannot_silently_use_row_padding(self):
        _, meta, _ = fixture(3)
        with self.assertRaisesRegex(ValueError, "unpacked"):
            shard_meta_for_dp(
                meta, dp_world=4, pad_to_dp=True, dynamic_batching_args={}
            )

    def test_actual_pg_loss_and_gradient_unchanged(self):
        for token_level in (True, False):
            with self.subTest(token_level=token_level):
                source, meta, client = fixture(32)
                loss_fn = ClippedPGLossFn(
                    ClippedPGLossConfig(
                        token_level_loss=token_level,
                        use_importance_sampling_correction=True,
                    )
                )
                original = (
                    -0.7 + torch.arange(128).reshape(32, 4) * 0.0001
                ).requires_grad_()
                loss, _ = loss_fn(
                    next_token_logprobs=original,
                    data=source,
                    global_valid_seqs=torch.tensor(32.0),
                    global_valid_toks=torch.tensor(128.0),
                )
                loss.backward()
                shards, _ = shard_meta_for_dp(meta, dp_world=64, pad_to_dp=True)
                gradient = torch.zeros_like(original)
                total_loss = 0.0
                for shard in shards:
                    row = int(shard.sample_ids[0][3:])
                    data = Worker(client)._fetch(shard)
                    local = original.detach()[row : row + 1].clone().requires_grad_()
                    shard_loss, metrics = loss_fn(
                        next_token_logprobs=local,
                        data=data,
                        global_valid_seqs=torch.tensor(32.0),
                        global_valid_toks=torch.tensor(128.0),
                    )
                    shard_loss.backward()
                    if shard.tags[0][DP_PADDING_TAG]:
                        torch.testing.assert_close(local.grad, torch.zeros_like(local))
                        self.assertEqual(float(metrics["num_valid_samples"]), 0)
                    else:
                        gradient[row] += local.grad[0]
                    total_loss += float(shard_loss.detach())
                torch.testing.assert_close(
                    gradient, original.grad, atol=1e-7, rtol=1e-6
                )
                self.assertAlmostEqual(total_loss, float(loss.detach()), places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
