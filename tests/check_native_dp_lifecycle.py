"""CPU checks of native Automodel batch admission; no optimizer update claim."""

import unittest
import copy
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from nemo_rl.data_plane.schema import DP_PADDING_TAG
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.policy.tq_policy import TQPolicy
from nemo_rl.models.policy.workers.dtensor_policy_worker_v2 import (
    DTensorPolicyWorkerV2Impl,
)


class LifecycleTests(unittest.TestCase):
    def test_padding_admission_is_limited_to_native_omni_microbatch_one(self):
        config = {
            "dtensor_cfg": {"enabled": True, "automodel_kwargs": {"force_hf": False}},
            "hf_config_overrides": {
                "architectures": ["NemotronH_Nano_Omni_Reasoning_V3"]
            },
            "train_micro_batch_size": 1,
        }
        self.assertTrue(TQPolicy._allows_dp_padding(SimpleNamespace(cfg=config)))
        variants = []
        for key, value in (
            ("dtensor_cfg", {"enabled": False}),
            ("dtensor_cfg", {"enabled": True, "automodel_kwargs": {"force_hf": True}}),
            ("dtensor_cfg", {"enabled": True}),
            ("hf_config_overrides", {"architectures": ["OtherModel"]}),
            ("hf_config_overrides", {}),
            ("train_micro_batch_size", 2),
        ):
            candidate = copy.deepcopy(config)
            candidate[key] = value
            variants.append(candidate)
        for candidate in variants:
            with self.subTest(config=candidate):
                self.assertFalse(
                    TQPolicy._allows_dp_padding(SimpleNamespace(cfg=candidate))
                )

    def worker(self):
        worker = object.__new__(DTensorPolicyWorkerV2Impl)
        worker._buffered_train_step = None
        worker.dp_size = 64
        worker.enable_seq_packing = False
        worker.cfg = {
            "train_global_batch_size": 32,
            "train_micro_batch_size": 1,
            "dynamic_batching": {"enabled": False},
        }
        worker.timer = SimpleNamespace(start=lambda name: None)
        return worker

    def row(self):
        return BatchedDataDict(
            {
                "input_ids": torch.ones(1, 5, dtype=torch.long),
                "sample_mask": torch.zeros(1),
                "token_mask": torch.zeros(1, 5),
                DP_PADDING_TAG: torch.tensor([True]),
            }
        )

    def test_one_full_dispatch_preserves_logical_gbs(self):
        worker = self.worker()
        worker.train = Mock(return_value={"synthetic": True})
        loss = object()
        worker.begin_train_step(loss, gbs=32, mbs=1)
        worker.train_microbatch(self.row())
        worker.finish_train_step()
        self.assertEqual(worker.train.call_count, 1)
        self.assertEqual(worker.train.call_args.kwargs["gbs"], 32)
        self.assertIsNone(worker._buffered_train_step)

    def test_partial_or_extra_dispatch_rejected(self):
        worker = self.worker()
        worker.begin_train_step(object(), gbs=32, mbs=1)
        with self.assertRaisesRegex(ValueError, "expected 1 local rows"):
            worker.train_microbatch(BatchedDataDict({"input_ids": torch.ones(2, 5)}))
        worker.train_microbatch(self.row())
        with self.assertRaisesRegex(ValueError, "exactly one"):
            worker.train_microbatch(self.row())

    def test_unsupported_microbatch_rejected(self):
        worker = self.worker()
        with self.assertRaisesRegex(ValueError, "microbatch size1"):
            worker.begin_train_step(object(), gbs=32, mbs=2)

    def test_unmasked_or_unstamped_data_rejected_before_cuda(self):
        for corruption in ("no_metadata", "sample_mask", "token_mask"):
            with self.subTest(corruption=corruption):
                worker = self.worker()
                data = self.row()
                if corruption == "no_metadata":
                    data.pop(DP_PADDING_TAG)
                    expected = "requires dispatcher"
                else:
                    data[corruption].fill_(1)
                    expected = "zero sample and token masks"
                with (
                    patch("torch.cuda.nvtx.range_push"),
                    patch("torch.cuda.nvtx.range_pop"),
                    self.assertRaisesRegex(ValueError, expected),
                ):
                    worker.train(data, object(), gbs=32, mbs=1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
