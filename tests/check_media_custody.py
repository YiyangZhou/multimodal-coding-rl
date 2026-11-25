"""Exercise media readback before immediate cleanup, including corrupt queues."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch
from nemo_gym.token_id_capture.staging.digest import hash_token_ids
from mmrl.media_custody import (acceptance_job_identity, observe_group,
                               maybe_observe_published_group, write_receipt)


def fixture(image=True, child=False):
    root_ids, ids = [11, 12], [11, 12, 13] if child else [11, 12]
    rollout = "sample_a0"

    def item(call, tokens, parent=None):
        return NS(staging_key=rollout + "/" + call, media_present=image,
                  snapshot=NS(cumulative_hash=hash_token_ids(tokens), rollout_id=rollout,
                    model_call_id=call, parent_call_id=parent, digest=call * 64,
                    weight_version=0, prev_len=2 if parent else 0, cum_len=len(tokens)))

    items = [item("a", root_ids)]
    if child:
        items.append(item("b", ids, "a"))
    part = NS(imgs=torch.arange(8, dtype=torch.float32).reshape(1, 2, 4),
              imgs_sizes=torch.tensor([[1, 2]], dtype=torch.int64), num_frames=None)
    part2 = NS(imgs=part.imgs + 10, imgs_sizes=part.imgs_sizes, num_frames=None)
    parts = {items[0].staging_key: part}
    if child:
        parts[items[1].staging_key] = part2
    data = {"input_ids": torch.tensor([ids]), "input_lengths": torch.tensor([len(ids)]),
            "sample_mask": torch.tensor([1.0]),
            "media_token_validity_mask": torch.zeros((1, len(ids)), dtype=torch.bool)}
    tags = {"multimodal_wire_fields": ["media_token_validity_mask"]}
    if image:
        selected = list(parts.values())
        data.update(pixel_values=torch.cat([v.imgs for v in selected], dim=1),
                    imgs_sizes=torch.cat([v.imgs_sizes for v in selected]).to(torch.int32).unsqueeze(0),
                    num_frames=torch.ones((1, len(selected)), dtype=torch.int32))
    row = NS(valid=True, rollout_id=rollout, token_ids=ids, staging_keys=[v.staging_key for v in items])

    def get(method, **kwargs):
        assert method == "get_samples" and kwargs["sample_ids"] == ["sample"]
        return {name: data[name] for name in kwargs["select_fields"]}

    source = NS(fetch_for_finalization=lambda keys: list(items),
                fetch_media=lambda values: [parts[v.staging_key] for v in values])
    finalizer = NS(_source=source, _partition_id="rollout_data", _call_dp=get)
    return finalizer, [row], ["sample"], [tags], items, data


class CustodyTests(unittest.TestCase):
    def test_immediate_cleanup_retains_durable_evidence(self):
        f, rows, ids, tags, items, data = fixture(child=True)
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
                "MMRL_MEDIA_CUSTODY_DIR": directory, "SLURM_JOB_ID": "123"}, clear=True):
            maybe_observe_published_group(f, "group", rows, ids, tags)
            items.clear()
            data.clear()
            paths = list(Path(directory).glob("*.json"))
            self.assertEqual(len(paths), 1)
            report = json.loads(paths[0].read_text())
            self.assertEqual(report["job"], 123)
            self.assertEqual(report["rows"]["sample"]["media_calls"], 2)
            self.assertTrue(report["rows"]["sample"]["pixel_and_geometry_bytes_equal"])
            self.assertNotIn("token_ids", paths[0].read_text())
            self.assertEqual(paths[0].stat().st_mode & 0o777, 0o600)

    def test_disabled_does_not_access_queue(self):
        with patch.dict(os.environ, {}, clear=True):
            maybe_observe_published_group(None, None, None, None, None)

    def test_durable_identity_without_slurm_variables(self):
        f, rows, ids, tags, _, _ = fixture()
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
                "MMRL_MEDIA_CUSTODY_DIR": directory, "OCI_JOB_ID": "456"}, clear=True):
            maybe_observe_published_group(f, "group", rows, ids, tags)
            report = json.loads(next(Path(directory).glob("*.json")).read_text())
            self.assertEqual(report["job"], 456)

    def test_consistent_identities(self):
        with patch.dict(os.environ, {"OCI_JOB_ID": "456", "SLURM_JOB_ID": "456"}, clear=True):
            self.assertEqual(acceptance_job_identity(), 456)

    def test_conflicting_identities_rejected_before_queue_access(self):
        with patch.dict(os.environ, {"MMRL_MEDIA_CUSTODY_DIR": "/unused",
                "OCI_JOB_ID": "456", "SLURM_JOB_ID": "123"}, clear=True):
            with self.assertRaisesRegex(ValueError, "Conflicting"):
                maybe_observe_published_group(None, None, None, None, None)

    def test_missing_or_invalid_identity_rejected(self):
        for values in ({}, {"OCI_JOB_ID": ""}, {"OCI_JOB_ID": "unknown"},
                       {"OCI_JOB_ID": "0"}, {"OCI_JOB_ID": "-1"},
                       {"OCI_JOB_ID": "１２３"},
                       {"OCI_JOB_ID": "123", "SLURM_JOB_ID": "invalid"}):
            with self.subTest(values=values), patch.dict(os.environ, values, clear=True):
                with self.assertRaisesRegex(ValueError, "explicit Slurm job identity"):
                    acceptance_job_identity()

    def test_corrupt_pixels_rejected(self):
        f, rows, ids, tags, _, data = fixture()
        data["pixel_values"] = data["pixel_values"] + 1
        with self.assertRaisesRegex(ValueError, "differs"):
            observe_group(f, "group", rows, ids, tags)

    def test_extra_pixels_rejected(self):
        f, rows, ids, tags, _, data = fixture()
        data["pixel_values"] = torch.cat([data["pixel_values"]] * 2, dim=1)
        with self.assertRaisesRegex(ValueError, "suffix"):
            observe_group(f, "group", rows, ids, tags)

    def test_corrupt_geometry_rejected(self):
        f, rows, ids, tags, _, data = fixture()
        data["imgs_sizes"] = data["imgs_sizes"] + 1
        with self.assertRaisesRegex(ValueError, "differs"):
            observe_group(f, "group", rows, ids, tags)

    def test_wrong_tokens_rejected(self):
        f, rows, ids, tags, _, data = fixture()
        data["input_ids"] = data["input_ids"] + 1
        with self.assertRaisesRegex(ValueError, "token IDs"):
            observe_group(f, "group", rows, ids, tags)

    def test_missing_parent_rejected(self):
        f, rows, ids, tags, items, _ = fixture(child=True)
        items[1].snapshot.parent_call_id = "missing"
        with self.assertRaises(KeyError):
            observe_group(f, "group", rows, ids, tags)

    def test_text_only_requires_actual_empty_mask(self):
        f, rows, ids, tags, _, data = fixture(image=False)
        result = observe_group(f, "group", rows, ids, tags)
        self.assertTrue(result["rows"]["sample"]["text_only_queue_media_verified"])
        data["media_token_validity_mask"][0, 0] = True
        with self.assertRaisesRegex(ValueError, "image positions"):
            observe_group(f, "group", rows, ids, tags)

    def test_invalid_row_does_not_claim_media_success(self):
        f, rows, ids, tags, _, _ = fixture()
        rows[0].valid = False
        result = observe_group(f, "group", rows, ids, tags)
        self.assertEqual(result["rows"], {})
        self.assertEqual(result["invalid_samples"], ["sample"])

    def test_immutable_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            value = {"job": 1, "group_id": "g", "rollout_ids": ["r"], "rows": {}}
            path = write_receipt(directory, value)
            self.assertEqual(write_receipt(directory, value), path)
            with self.assertRaisesRegex(ValueError, "changed"):
                write_receipt(directory, value | {"rows": {"fabricated": True}})


if __name__ == "__main__":
    unittest.main()
