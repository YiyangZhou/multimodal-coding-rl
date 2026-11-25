"""Exercise the native HTTP renderer's remaining-context budget without model inference."""

import argparse
import asyncio
import hashlib
import importlib.util
import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


async def check(root):
    fixture_path = root / "tests/unit/models/generation/test_vllm_generation.py"
    spec = importlib.util.spec_from_file_location(
        "native_generation_budget_fixtures", fixture_path
    )
    fixture = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = fixture
    spec.loader.exec_module(fixture)
    reports = []
    with pytest.MonkeyPatch.context() as monkeypatch:
        _, _, serving = fixture._install_fake_vllm_openai_modules(monkeypatch)
        worker = fixture.VllmAsyncGenerationWorkerImpl.__new__(
            fixture.VllmAsyncGenerationWorkerImpl
        )
        worker.cfg = {
            "temperature": 1.0,
            "top_p": 1.0,
            "val_temperature": 0.0,
            "val_top_p": 1.0,
            "vllm_cfg": {"cap_max_tokens_to_context": True},
        }
        worker.llm = MagicMock(
            model_config=SimpleNamespace(max_model_len=262144), renderer=MagicMock()
        )
        worker._http_engine_client = worker.llm
        worker._capture_calls = {}
        worker.token_capture = None
        worker._capture_media = False
        worker.llm_async_engine_args = MagicMock()
        worker.llm_async_engine_args.create_model_config.return_value = MagicMock(
            served_model_name="policy", model="unused"
        )
        app = fixture._FakeFastAPIApp()
        worker._setup_vllm_openai_api_server(app)
        renderer = serving.instances[0].kwargs["online_renderer"]
        renderer.model_config = SimpleNamespace(max_model_len=262144)
        handler = next(fn for path, fn in app.routes if path == "/v1/chat/completions")
        request_type = inspect.signature(handler).parameters["request"].annotation
        base_renderer = sys.modules["vllm.renderers.online_renderer"].OnlineRenderer
        for field, requested, input_length in (
            ("max_tokens", 262144, 24000),
            ("max_completion_tokens", 262144, 131072),
            ("max_tokens", 262144, 262143),
            ("max_tokens", 32768, 24000),
            ("max_tokens", 262144, 262144),
        ):
            tokens = [11] * input_length
            messages = [
                {"role": "user", "content": "Synthetic already-tokenized prompt"}
            ]
            request = request_type()
            request.max_tokens = requested if field == "max_tokens" else 17
            request.max_completion_tokens = (
                requested if field == "max_completion_tokens" else None
            )
            request.required_prefix_token_ids = None
            request.ng_capture = None
            seen = []

            async def preprocess(self, **kwargs):
                # This is the vLLM boundary BEFORE prompt tokenization/validation.
                seen.append(getattr(kwargs["request"], field))
                assert seen[-1] == 1
                assert kwargs["messages"] == messages
                return (messages, [{"prompt_token_ids": tokens}])

            monkeypatch.setattr(
                base_renderer, "preprocess_chat", preprocess, raising=False
            )
            call = renderer.preprocess_chat(request, messages, None, "openai", {})
            if input_length >= 262144:
                with pytest.raises(ValueError, match="context length"):
                    await call
                reports.append(
                    {
                        "field": field,
                        "input_tokens": input_length,
                        "overflow_rejected": True,
                    }
                )
            else:
                result = await call
                expected = min(requested, 262144 - input_length)
                assert getattr(request, field) == expected
                assert result[1][0]["prompt_token_ids"] is tokens
                assert result[0] == messages
                assert seen == [1]
                reports.append(
                    {
                        "field": field,
                        "requested": requested,
                        "input_tokens": input_length,
                        "effective_output_tokens": expected,
                        "history_preserved": True,
                    }
                )
    return reports


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rl-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = asyncio.run(check(args.rl_root))
    relative = "nemo_rl/models/generation/vllm/vllm_worker_async.py"
    result = {
        "passed": True,
        "model_inference": False,
        "native_preprocess": True,
        "tokenization_fixture": True,
        "reports": reports,
        "source_sha256": {
            relative: hashlib.sha256((args.rl_root / relative).read_bytes()).hexdigest()
        },
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
