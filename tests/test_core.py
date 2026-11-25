import difflib
import io
import math
import unittest
from unittest.mock import AsyncMock, patch

from mmrl.artifacts import apply_diff, extract_answer
from mmrl.config import gym_config, training_config
from mmrl.reward import QualityJudge, InfrastructureError, Quality
from mmrl.sandbox import validate_endpoint, validate_image, image_pull_options
from mmrl.service import CodingService
from mmrl.tasks import Task, route, gym_row


class CoreTests(unittest.TestCase):
    def test_process_composition_is_explicit_and_pairs_service_with_learner(self):
        from mmrl.process_rewards import ProcessRewardConfig
        process = ProcessRewardConfig(progress_weight=.04)
        image = "image@sha256:" + "a" * 64
        ordinary = gym_config(image, "auto")
        enabled = gym_config(image, "auto", process_reward_config=process)
        self.assertEqual(ordinary["mm_coding"]["resources_servers"]["mm_coding"]["entrypoint"], "app.py")
        resource = enabled["mm_coding"]["resources_servers"]["mm_coding"]
        self.assertEqual(resource["entrypoint"], "process_app.py")
        self.assertEqual(resource["process_reward_config"]["progress_weight"], .04)
        with self.assertRaisesRegex(ValueError, "managed capture gateway"):
            gym_config(image, "https://capture.example", process_reward_config=process)
        base = {"data_plane": {"enabled": True}, "async_rl": {},
                "policy": {"generation": {"temperature": .8, "top_p": .9, "top_k": None}},
                "data": {"max_input_seq_length": 32768}, "grpo": {"adv_estimator": {"name": "grpo"}},
                "checkpointing": {}, "logger": {}}
        normal = training_config(base, "/gym.json", "/data.jsonl", "/run")
        cfg = training_config(base, "/gym.json", "/data.jsonl", "/run", process_reward_config=process)
        self.assertNotIn("process_returns", normal["token_capture"])
        self.assertEqual(cfg["token_capture"]["process_returns"]["binder"], "mmrl.process_capture.bind_native_returns")
        self.assertEqual(base["grpo"]["adv_estimator"]["name"], "grpo")
        base["grpo"]["adv_estimator"]["name"] = "gdpo"
        with self.assertRaisesRegex(ValueError, "require GRPO"):
            training_config(base, "/gym.json", "/data.jsonl", "/run", process_reward_config=process)

    def test_private_image_credentials_do_not_follow_foreign_registry(self):
        with patch.dict(
            "os.environ",
            {
                "REGISTRY_HOST": "registry.example:5005",
                "REGISTRY_USERNAME": "test-user",
                "REGISTRY_PASSWORD": "test-only-password",
            },
            clear=True,
        ):
            self.assertEqual(
                image_pull_options("docker.io/library/python@sha256:" + "a" * 64), {}
            )
            self.assertEqual(
                image_pull_options(
                    "registry.example:5005/group/image@sha256:" + "a" * 64
                ),
                {
                    "image_auth": {
                        "username": "test-user",
                        "password": "test-only-password",
                    }
                },
            )

    def test_partial_registry_configuration_fails_without_echoing_secrets(self):
        with patch.dict(
            "os.environ", {"REGISTRY_PASSWORD": "test-only-password"}, clear=True
        ):
            with self.assertRaisesRegex(ValueError, "REGISTRY_HOST") as error:
                image_pull_options("registry.example/group/image@sha256:" + "a" * 64)
            self.assertNotIn("test-only-password", str(error.exception))

    def setUp(self):
        self.task = Task("mmsvg_text2svg", "example", "Draw a red circle")

    def test_quality_formula(self):
        self.assertAlmostEqual(Quality(0.8, 0.6).reward, 0.76)
        for value in (math.nan, math.inf, -1, 1.1, True):
            with self.assertRaises(ValueError):
                Quality(value, 0.5)

    def test_siblings_share_route_and_epoch_is_reproducible(self):
        self.assertEqual(
            len({route(self.task, "mixed", epoch=3) for _ in range(16)}), 1
        )
        self.assertEqual(route(self.task, "mixed", multi_fraction=0), "single")
        self.assertIn(
            route(self.task, "mixed", multi_fraction=1), ("codex", "opencode", "hermes")
        )

    def test_mixed_contains_every_branch(self):
        routes = {
            route(Task("mmsvg_text2svg", str(i), "Circle"), "mixed", multi_fraction=0.5)
            for i in range(1000)
        }
        self.assertEqual(routes, {"single", "opencode", "codex", "hermes"})

    def test_invalid_route(self):
        with self.assertRaises(ValueError):
            route(self.task, "mixed", harnesses=("unknown",))
        with self.assertRaises(ValueError):
            route(self.task, "mixed", multi_fraction=math.nan)

    def test_no_text_reference_or_missing_visual_reference(self):
        with self.assertRaises(ValueError):
            Task("mmsvg_text2svg", "x", "Circle", ["image.png"])
        with self.assertRaises(ValueError):
            Task("chart2code", "x", "Chart")

    def test_only_complete_answer(self):
        with self.assertRaises(ValueError):
            extract_answer("<think>still thinking", "svg")
        self.assertEqual(
            extract_answer("<think>done</think>\n```svg\n<svg/>\n```", "svg"), "<svg/>"
        )

    def test_strict_diff_and_no_partial_fallback(self):
        before, after = "a\nb\nc\n", "a\nB\nc\n"
        diff = "".join(
            difflib.unified_diff(
                before.splitlines(True),
                after.splitlines(True),
                "a/page.html",
                "b/page.html",
            )
        )
        self.assertEqual(apply_diff(before, diff), after)
        with self.assertRaises(ValueError):
            apply_diff(before, diff.replace("-b\n", "-wrong\n"))
        with self.assertRaises(ValueError):
            apply_diff(before, "<html>fragment fallback</html>")

    def test_endpoint_never_defaults_local(self):
        for value in (
            "",
            "http://localhost:8080",
            "https://127.0.0.1",
            "https://host/v1",
            "https://user:pass@host",
        ):
            with self.assertRaises(ValueError):
                validate_endpoint(value)
        self.assertEqual(
            validate_endpoint("https://assigned.example/"), "https://assigned.example"
        )

    def test_real_harness_names_and_remote_only(self):
        cfg = gym_config("image@sha256:" + "a" * 64, "https://capture.example")
        for name in ("opencode", "codex", "hermes"):
            adapter = cfg[f"mm_{name}"]["responses_api_agents"]["harness_agent"]
            self.assertEqual(adapter["agent"], name)
            self.assertEqual(adapter["sandbox_provider"], "sandbox")
            self.assertTrue(adapter["runner_failure_as_response"])

    def test_reject_invalid_deployment_before_starting(self):
        image = "image@sha256:" + "a" * 64
        for value in ("image:latest", "image@sha256:short"):
            with self.assertRaises(ValueError):
                validate_image(value)
        with self.assertRaises(ValueError):
            gym_config(image, "https://localhost")
        with self.assertRaises(ValueError):
            gym_config(image, "https://capture.example", concurrency=0)
        with self.assertRaises(ValueError):
            gym_config(
                image,
                "https://capture.example",
                max_output_tokens=131072,
                context_window=65536,
            )

    def test_large_budget_survives_dataset_and_harness_composition(self):
        import json
        import tempfile
        from pathlib import Path
        from mmrl.tasks import write_rows

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.jsonl"
            write_rows([self.task], path, "multi", max_output_tokens=262144)
            self.assertEqual(
                json.loads(path.read_text())["responses_create_params"][
                    "max_output_tokens"
                ],
                262144,
            )
        cfg = gym_config(
            "image@sha256:" + "a" * 64,
            "https://capture.example",
            max_output_tokens=262144,
            context_window=262144,
        )
        opencode = cfg["mm_opencode"]["responses_api_agents"]["harness_agent"][
            "agent_kwargs"
        ]
        self.assertEqual(
            opencode["env"]["OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX"], "262144"
        )
        self.assertEqual(
            opencode["opencode_config"]["compaction"], {"auto": False, "prune": False}
        )
        self.assertEqual(
            opencode["opencode_config"]["provider"]["gym"]["models"]["policy"]["limit"][
                "output"
            ],
            262144,
        )

    def test_composition_isolates_native_capture_from_donor_run(self):
        base = {
            "data_plane": {"enabled": True},
            "policy": {"generation": {"temperature": 0.8, "top_p": 0.9, "top_k": None}},
            "async_rl": {},
            "data": {"max_input_seq_length": 262144},
            "grpo": {},
            "checkpointing": {},
            "logger": {},
            "token_capture": {
                "enabled": True,
                "capture_dir": "/old/captures",
                "control_auth_token": "test-old-token",
            },
        }
        cfg = training_config(base, "/new/gym.json", "/new/data.jsonl", "/new/run")
        self.assertEqual(cfg["token_capture"]["capture_dir"], "/new/run/captures")
        self.assertNotIn("control_auth_token", cfg["token_capture"])
        self.assertFalse(cfg["env"]["nemo_gym"]["token_id_capture"]["rebuild_response"])
        self.assertEqual(base["token_capture"]["capture_dir"], "/old/captures")
        pin = cfg["env"]["nemo_gym"]["policy_model"]["responses_api_models"][
            "vllm_model"
        ]["sampling_overrides"]
        self.assertEqual(pin, {"temperature": 0.8, "top_p": 0.9, "top_k": -1})
        self.assertIsNone(base["policy"]["generation"]["top_k"])
        base["policy"]["generation"]["top_k"] = 50
        with self.assertRaisesRegex(ValueError, "disabled top-k"):
            training_config(base, "/new/gym.json", "/new/data.jsonl", "/new/run")

    def test_missing_renderer_is_infrastructure_failure(self):
        import asyncio
        from contextlib import asynccontextmanager
        from types import SimpleNamespace
        from mmrl.sandbox import RemoteSandbox

        provider = AsyncMock()
        provider.exec.return_value = SimpleNamespace(error_type=None, return_code=1)
        sandbox = RemoteSandbox("image@sha256:" + "a" * 64, provider=provider)

        @asynccontextmanager
        async def session():
            yield "probe"

        sandbox.session = session
        sandbox.prepare = AsyncMock()
        with self.assertRaises(InfrastructureError):
            asyncio.run(sandbox.render("mmsvg_text2svg", "<svg/>"))

    def test_single_code_edit_receives_source_not_reference_patch(self):
        task = Task(
            "webmmu_code_edit",
            "edit",
            "Change the heading",
            original_source="<h1>Old</h1>",
            reference_patch="private-answer",
        )
        content = gym_row(task, "single")["responses_create_params"]["input"]
        self.assertIn(task.original_source, str(content))
        self.assertNotIn(task.reference_patch, str(content))

    def test_single_code_edit_preserves_embedded_source_without_duplication(self):
        source = "<html><h1>Old</h1></html>"
        prompt = "Change the heading. Original page:\n```html\n" + source + "\n```"
        task = Task("webmmu_code_edit", "embedded", prompt,
                    original_source=source, reference_patch="private-reference-diff")
        row = gym_row(task, "single")
        content = row["responses_create_params"]["input"][-1]["content"]
        text = "\n".join(item["text"] for item in content if item["type"] == "input_text")
        self.assertEqual(text, prompt)
        self.assertEqual(text.count(source), 1)
        self.assertNotIn(task.reference_patch, text)
        self.assertEqual(row["task"]["original_source"], source)
        self.assertEqual(row["task"]["reference_patch"], task.reference_patch)


class RewardTests(unittest.IsolatedAsyncioTestCase):
    async def test_malformed_api_response_is_infrastructure_failure(self):
        with patch.dict("os.environ", {"OPENAI_API_KEYS": "test-only"}):
            judge = QualityJudge()
        for body in (b"not-json", b"[]"):
            with patch("urllib.request.urlopen", return_value=io.BytesIO(body)):
                with self.assertRaises(InfrastructureError):
                    await judge.grade(
                        Task("mmsvg_text2svg", "x", "Circle"),
                        "<svg/>",
                        b"\x89PNG\r\n\x1a\n",
                        [],
                    )

    async def test_execution_failure_never_calls_judge(self):
        sandbox, judge = AsyncMock(), AsyncMock()
        sandbox.render.return_value = None
        result = await CodingService(sandbox, judge).score(
            Task("mmsvg_text2svg", "x", "Circle"), "<svg/>"
        )
        self.assertEqual(result["reward"], 0.0)
        judge.grade.assert_not_awaited()

    async def test_api_failure_is_not_model_zero(self):
        sandbox, judge = AsyncMock(), AsyncMock()
        sandbox.render.return_value = b"png"
        judge.grade.side_effect = InfrastructureError("budget unavailable")
        with self.assertRaises(InfrastructureError):
            await CodingService(sandbox, judge).score(
                Task("mmsvg_text2svg", "x", "Circle"), "<svg/>"
            )

    async def test_single_and_harness_use_same_reward(self):
        sandbox, judge = AsyncMock(), AsyncMock()
        sandbox.render.return_value = b"png"
        judge.grade.return_value = Quality(0.8, 0.6)
        service = CodingService(sandbox, judge)
        task = Task("mmsvg_text2svg", "x", "Circle")
        a = await service.score(task, "```svg\n<svg/>\n```")
        b = await service.score(task, "<svg/>", harness_artifact=True)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
