import difflib
import io
import json
import unittest
from unittest.mock import AsyncMock, patch

from PIL import Image

from mmrl.reward import QualityJudge, InfrastructureError, Quality
from mmrl.service import CodingService
from mmrl.tasks import Task, gym_row


def diff(before, after):
    return "".join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                        "a/original.html", "b/original.html"))


class EvidenceTests(unittest.IsolatedAsyncioTestCase):
    def test_legacy_judge_content_survives_without_entering_model_input(self):
        task = Task.from_legacy_context({
            "benchmark_id": "mmsvg_text2svg", "sample_id": "example",
            "prompt": "Draw a circle", "semantic_prompt": "One red circle",
            "ground_truth": "private-reference-svg", "target_image_paths": [],
        })
        self.assertEqual(task.semantic_prompt, "One red circle")
        self.assertEqual(task.ground_truth, "private-reference-svg")
        for route in ("single", "opencode", "codex", "hermes"):
            row = gym_row(task, route)
            self.assertEqual(row["task"]["ground_truth"], task.ground_truth)
            self.assertNotIn(task.ground_truth, json.dumps(row["responses_create_params"]))
        # Existing prepared rows remain readable; re-export is required to add
        # evidence discarded by older exports.
        self.assertEqual(Task("mmsvg_text2svg", "old", "Circle").ground_truth, "")

    async def test_api_distinguishes_original_target_and_candidate(self):
        task = Task("webmmu_code_edit", "x", "Change title", original_source="before",
                    reference_patch="patch", ground_truth="private-ref", semantic_prompt="Edit title")
        score = dict(quality=0.7, task_fidelity=0.8, visual_fidelity=0.6, reason="Title changed")
        response = {"status": "completed", "output": [{"type": "message", "content": [
            {"type": "output_text", "text": json.dumps(score)}]}]}
        stream = io.BytesIO()
        Image.new("RGB", (4, 4), "red").save(stream, format="PNG")
        png = stream.getvalue()
        with patch.dict("os.environ", {"OPENAI_API_KEYS": "test-only"}):
            judge = QualityJudge()
        with patch("urllib.request.urlopen", return_value=io.BytesIO(json.dumps(response).encode())) as call:
            await judge.grade(task, "candidate-diff", png + b"candidate", [png + b"input"],
                              original_render=png + b"original", target_render=png + b"target")
        content = json.loads(call.call_args.args[0].data)["input"][0]["content"]
        evidence = json.loads(content[0]["text"])
        self.assertEqual(evidence["reference_code_or_patch"], "private-ref")
        self.assertEqual(evidence["semantic_prompt"], "Edit title")
        self.assertTrue(evidence["reference_render_available"])
        self.assertEqual([x["text"] for x in content[1:] if x["type"] == "input_text"], [
            "ORIGINAL input screenshot BEFORE the requested edit (not the target)",
            "ORIGINAL page rendering before the edit", "REFERENCE target rendering",
            "CANDIDATE rendering to grade",
        ])

    async def test_code_edit_renders_actual_reference_and_candidate_separately(self):
        before, after = "<title>Old</title>\n", "<title>New</title>\n"
        patch_text = diff(before, after)
        task = Task("webmmu_code_edit", "x", "Change title", original_source=before,
                    reference_patch=patch_text)
        sandbox, judge = AsyncMock(), AsyncMock()
        sandbox.render.side_effect = [b"candidate", b"original", b"target"]
        judge.grade.return_value = Quality(0.8, 0.6)
        result = await CodingService(sandbox, judge).score(task, patch_text, harness_artifact=True)
        self.assertAlmostEqual(result["reward"], 0.76)
        # Restored legacy patch parser joins source lines without a terminal newline.
        expected_patched = "<title>New</title>"
        self.assertEqual([c.args[1] for c in sandbox.render.await_args_list],
                         [expected_patched, before, expected_patched])
        self.assertEqual(judge.grade.await_args.kwargs,
                         {"original_render": b"original", "target_render": b"target"})

    async def test_bad_reference_does_not_zero_valid_candidate(self):
        before, after = "a\n", "b\n"
        task = Task("webmmu_code_edit", "x", "Edit", original_source=before,
                    reference_patch="dataset partial diff")
        sandbox, judge = AsyncMock(), AsyncMock()
        sandbox.render.side_effect = [b"candidate", b"original"]
        judge.grade.return_value = Quality(0.8, 0.6)
        result = await CodingService(sandbox, judge).score(task, diff(before, after), harness_artifact=True)
        self.assertEqual(result["execution"], 1)
        self.assertEqual(judge.grade.await_args.kwargs, {"original_render": b"original"})

    async def test_original_render_failure_is_not_a_model_zero(self):
        task = Task("webmmu_code_edit", "x", "Edit", original_source="a\n",
                    reference_patch=diff("a\n", "b\n"))
        sandbox, judge = AsyncMock(), AsyncMock()
        sandbox.render.side_effect = [b"candidate", None]
        with self.assertRaises(InfrastructureError):
            await CodingService(sandbox, judge).score(task, task.reference_patch, harness_artifact=True)
        judge.grade.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
