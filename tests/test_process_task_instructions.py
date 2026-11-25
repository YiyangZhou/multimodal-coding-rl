"""Process preview must be reachable without leaking private judge evidence."""
import json
from pathlib import Path
import tempfile
import unittest

from mmrl.tasks import Task, gym_row, route, write_rows


class ProcessTaskInstructionsTests(unittest.TestCase):
    def test_process_tool_keeps_task_and_private_evidence_separate(self):
        task = Task("mmsvg_text2svg", "fixture", "Draw a red square.",
                    semantic_prompt="PRIVATE_JUDGE_INSTRUCTION", ground_truth="PRIVATE_ANSWER")
        for harness in ("opencode", "codex", "hermes"):
            with self.subTest(harness=harness):
                old = gym_row(task, harness, 262144)
                new = gym_row(task, harness, 262144, process_rewards=True)
                self.assertEqual(old["task"], new["task"])
                before, after = old["responses_create_params"], new["responses_create_params"]
                self.assertEqual(before["input"][1:], after["input"][1:])
                self.assertEqual(before | {"input": after["input"]}, after)
                instructions = after["input"][0]["content"]
                self.assertIn("mmrl MCP server", instructions)
                self.assertNotIn("/opt/mmrl/render.py", instructions)
                self.assertNotIn("PRIVATE_", json.dumps(after))
                self.assertIn("Save your final answer", instructions)

    def test_single_rows_unchanged_and_mixed_routes_stable(self):
        tasks = [Task("mmsvg_text2svg", str(i), "Draw a circle.") for i in range(20)]
        for task in tasks:
            self.assertEqual(gym_row(task, "single"), gym_row(task, "single", process_rewards=True))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mixed.jsonl"
            write_rows(tasks, path, "mixed", process_rewards=True)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertTrue(any(row["route"] == "single" for row in rows))
        self.assertTrue(any(row["route"] != "single" for row in rows))
        for task, row in zip(tasks, rows):
            self.assertEqual(row["route"], route(task, "mixed"))
            self.assertEqual(row, gym_row(task, row["route"], process_rewards=True))


if __name__ == "__main__":
    unittest.main()
