import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("media_summary", Path(__file__).resolve().parents[1] / "scripts/summarize_media_custody.py")
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "receipts").mkdir()
        (self.root / "lineage").mkdir()
        self.call = {"staging_digest": "digest", "model_call_id": "a", "parent_call_id": None,
                     "weight_version": 0, "prev_len": 0, "cum_len": 2, "media_present": True}
        self.ledger = self.call | {"staging_key": "r/a", "cumulative_hash": "tokens"}
        self.receipt = {"schema_version": 1, "job": 1, "observer_sha256": "observer",
                       "observation_boundary": "published_queue_before_finalizer_returns",
                       "calls": {"r/a": self.call}, "rows": {"sample": {
                           "terminal_call": "r/a", "rollout_id": "r", "token_hash": "tokens",
                           "captured_calls": 1, "media_calls": 1}}}

    def run_summary(self):
        (self.root / "lineage/a.lineage.jsonl").write_text(json.dumps(self.ledger) + "\n")
        (self.root / "receipts/a.json").write_text(json.dumps(self.receipt))
        return summary.summarize(self.root / "receipts", self.root / "lineage", job=1, observer_sha256="observer")

    def test_binds_receipt_and_lineage_hashes(self):
        _, result = self.run_summary()
        self.assertEqual(set(result["rows"]), {"sample"})
        self.assertEqual(len(result["evidence_sha256"]), 2)
        for name, expected in result["evidence_sha256"].items():
            self.assertEqual(summary.sha(Path(name)), expected)

    def test_wrong_job_rejected(self):
        self.receipt["job"] = 2
        with self.assertRaisesRegex(ValueError, "provenance"):
            self.run_summary()

    def test_changed_weight_version_rejected(self):
        self.ledger["weight_version"] = 1
        with self.assertRaisesRegex(ValueError, "weight_version"):
            self.run_summary()

    def test_changed_tokens_rejected(self):
        self.ledger["cumulative_hash"] = "changed"
        with self.assertRaisesRegex(ValueError, "tokens"):
            self.run_summary()

    def test_missing_chain_rejected(self):
        self.receipt["calls"] = {}
        with self.assertRaises(KeyError):
            self.run_summary()

    def test_no_receipts_rejected(self):
        with self.assertRaisesRegex(ValueError, "No synchronous"):
            summary.summarize(self.root / "receipts", self.root / "lineage", job=1, observer_sha256="observer")


if __name__ == "__main__":
    unittest.main()
