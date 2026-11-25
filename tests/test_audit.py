import json
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

from mmrl.audit import VerificationAudit
from mmrl.reward import Quality
from mmrl.service import CodingService
from mmrl.tasks import Task


class AuditTests(unittest.IsolatedAsyncioTestCase):
    async def test_harness_failure_codes_survive_without_arbitrary_error_text(self):
        codes = ("codex_timeout", "codex_nonzero_exit", "codex_launch_failure",
                 "codex_turn_incomplete_or_failed")
        with tempfile.TemporaryDirectory() as directory:
            for code in (*codes, "https://private.invalid/?token=secret", {"secret": "private"}, None):
                audit = VerificationAudit(
                    directory, task=Task("mmsvg_text2svg", "x", "Circle"),
                    route="codex", response_id="resp-test", status="failed", failure_code=code)
                audit.finish({"reward": 0.0, "mask_sample": True})
                text = (audit.path / "record.json").read_text()
                record = json.loads(text)
                self.assertNotIn("secret", text)
                self.assertNotIn("private", text)
                self.assertTrue(record["result"]["mask_sample"])
                expected = code if isinstance(code, str) and code in codes else "unclassified"
                if code is None:
                    self.assertNotIn("harness_failure_code", record)
                else:
                    self.assertEqual(record["harness_failure_code"], expected)

    async def test_evidence_preserves_reward_and_actual_render(self):
        task = Task("mmsvg_text2svg", "../../untrusted", "Circle")
        sandbox, judge = AsyncMock(), AsyncMock()
        sandbox.render.return_value = b"rendered-by-verifier"
        judge.grade.return_value = Quality(0.8, 0.6, "Correct object; minor geometry differences")
        service = CodingService(sandbox, judge)
        expected = await service.score(task, "<svg/>", harness_artifact=True)
        with tempfile.TemporaryDirectory() as directory:
            audit = VerificationAudit(directory, task=task, route="codex",
                                      response_id="resp-test", status="completed")
            actual = await service.score(task, "<svg/>", harness_artifact=True, audit=audit)
            self.assertEqual(actual, expected)
            audit.finish(actual | {"authorization": "must-not-be-saved", "failure_reason": "private-url"})
            record = json.loads((audit.path / "record.json").read_text())
            self.assertTrue(record["complete"])
            self.assertNotIn("authorization", record["result"])
            self.assertNotIn("failure_reason", record["result"])
            self.assertEqual((audit.path / "render.png").read_bytes(), b"rendered-by-verifier")
            self.assertEqual((audit.path / "answer.txt").read_text(), "<svg/>")
            self.assertEqual((audit.path / "judge_reason.txt").read_text(),
                             judge.grade.return_value.reason)
            self.assertEqual(audit.path.parent, Path(directory))
            for path in audit.path.iterdir():
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            with self.assertRaises(ValueError):
                audit.write("../outside", b"rejected")

    async def test_invalid_diff_keeps_original_answer_without_judging(self):
        task = Task("webmmu_code_edit", "x", "Change title", references=["unused.png"],
                    original_source="<html/>", reference_patch="reference diff")
        sandbox, judge = AsyncMock(), AsyncMock()
        with tempfile.TemporaryDirectory() as directory:
            audit = VerificationAudit(directory, task=task, route="opencode",
                                      response_id="resp-test", status="completed")
            answer = "<html>not a unified diff</html>"
            result = await CodingService(sandbox, judge).score(
                task, answer, harness_artifact=True, audit=audit)
            audit.finish(result)
            self.assertEqual(result["failure_kind"], "mm_coding:invalid_artifact")
            self.assertEqual((audit.path / "answer.txt").read_text(), answer)
            self.assertFalse((audit.path / "render.png").exists())
            sandbox.render.assert_not_awaited()
            judge.grade.assert_not_awaited()

    async def test_unfinished_audit_cannot_be_mistaken_for_completed_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            audit = VerificationAudit(directory, task=Task("mmsvg_text2svg", "x", "Circle"),
                                      route="hermes", response_id="resp-test", status="failed")
            self.assertFalse(json.loads((audit.path / "record.json").read_text())["complete"])


if __name__ == "__main__":
    unittest.main()
