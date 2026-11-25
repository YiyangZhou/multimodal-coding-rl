"""Native Gym request conversion and engine-bound sampling; no model inference."""

import argparse
import copy
import json
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient
from mmrl.config import training_config
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.responses_converter import ResponsesConverter
from nemo_gym.server_utils import ServerClient
from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig


class ProtocolTests(unittest.TestCase):
    wire_request = None
    codex_message_requests = None
    opencode_requests = None
    hermes_requests = None
    codex_image_requests = None

    def setUp(self):
        base = {
            "data_plane": {"enabled": True},
            "async_rl": {},
            "policy": {"generation": {"temperature": 0.8, "top_p": 0.9, "top_k": None}},
            "data": {"max_input_seq_length": 262144},
            "grpo": {},
            "checkpointing": {},
            "logger": {},
        }
        cfg = training_config(base, "/tmp/gym.json", "/tmp/tasks.jsonl", "/tmp/run")
        settings = cfg["env"]["nemo_gym"]["policy_model"]["responses_api_models"][
            "vllm_model"
        ]
        self.pin = settings["sampling_overrides"]
        config = VLLMModelConfig(
            host="127.0.0.1",
            port=8000,
            entrypoint="app.py",
            name="protocol",
            base_url="http://unused.invalid/v1",
            api_key="synthetic-placeholder",
            model="policy",
            return_token_id_information=False,
            uses_reasoning_parser=True,
            **settings,
        )
        self.model = VLLMModel(
            config=config,
            server_client=MagicMock(spec=ServerClient, global_config_dict={}),
        )
        self.backend = MagicMock()
        self.backend.create_chat_completion = AsyncMock(
            return_value={
                "id": "chatcmpl-protocol",
                "object": "chat.completion",
                "created": 0,
                "model": "policy",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "logprobs": None,
                        "message": {"role": "assistant", "content": "OK"},
                    }
                ],
                "usage": {
                    "prompt_tokens": 2,
                    "completion_tokens": 1,
                    "total_tokens": 3,
                },
            }
        )
        self.model._clients = [self.backend]
        self.client = TestClient(self.model.setup_webserver())
        self.converter = ResponsesConverter(return_token_id_information=False)

    def check_request(self, path, body):
        self.backend.create_chat_completion.reset_mock()
        response = self.client.post(path, json=body)
        self.assertEqual(response.status_code, 200, response.text[:500])
        self.backend.create_chat_completion.assert_called_once()
        outbound = self.backend.create_chat_completion.call_args.kwargs
        self.assertEqual({k: outbound[k] for k in self.pin}, self.pin)
        self.assertTrue(outbound["chat_template_kwargs"]["enable_thinking"])
        return outbound

    def test_direct_chat_missing_sampling(self):
        self.check_request(
            "/v1/chat/completions", {"messages": [{"role": "user", "content": "OK?"}]}
        )

    def test_direct_chat_conflicting_sampling(self):
        self.check_request(
            "/v1/chat/completions",
            {
                "messages": [{"role": "user", "content": "OK?"}],
                "temperature": 0.2,
                "top_p": 0.3,
                "metadata": {
                    "extra_body": json.dumps({"temperature": 0.4, "top_p": 0.5})
                },
            },
        )

    def test_responses_missing_sampling(self):
        self.check_request(
            "/v1/responses", {"input": "OK?", "include": [], "stream": False}
        )

    def test_responses_stream_missing_sampling(self):
        self.check_request(
            "/v1/responses", {"input": "OK?", "include": [], "stream": True}
        )

    def test_empty_include_equals_absent(self):
        base = {"input": "OK?"}

        def convert(body):
            return self.converter.responses_to_chat_completion_create_params(
                NeMoGymResponseCreateParamsNonStreaming.model_validate(body)
            ).model_dump()

        self.assertEqual(convert(base), convert(base | {"include": []}))
        self.assertEqual(convert(base), convert(base | {"include": None}))

    def test_nonempty_include_still_rejected(self):
        for field in ("reasoning.encrypted_content", "message.output_text.logprobs"):
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(NotImplementedError, "include"),
            ):
                self.converter.responses_to_chat_completion_create_params(
                    NeMoGymResponseCreateParamsNonStreaming(
                        input="OK?", include=[field]
                    )
                )

    def test_stateful_responses_still_rejected(self):
        with self.assertRaisesRegex(NotImplementedError, "previous_response_id"):
            self.converter.responses_to_chat_completion_create_params(
                NeMoGymResponseCreateParamsNonStreaming(
                    input="OK?", include=[], previous_response_id="resp_previous"
                )
            )

    def test_real_codex_request(self):
        if self.wire_request is None:
            self.skipTest("Supply --codex-request from the actual pinned CLI probe")
        request = copy.deepcopy(self.wire_request)
        self.assertEqual(request["include"], [])
        outbound = self.check_request("/v1/responses", request)
        self.assertNotIn("include", outbound)
        self.assertTrue(outbound["tools"])

    def test_tool_image_reaches_engine(self):
        image = "data:image/png;base64,iVBORw0KGgo="
        for stream in (False, True):
            with self.subTest(stream=stream):
                outbound = self.check_request(
                    "/v1/responses",
                    {
                        "stream": stream,
                        "input": [
                            {"role": "user", "content": "Inspect the rendered code."},
                            {
                                "type": "function_call",
                                "call_id": "call_render",
                                "name": "render",
                                "arguments": "{}",
                            },
                            {
                                "type": "function_call_output",
                                "call_id": "call_render",
                                "output": [
                                    {"type": "input_text", "text": "Render complete"},
                                    {
                                        "type": "input_image",
                                        "image_url": image,
                                        "detail": "high",
                                    },
                                ],
                            },
                        ],
                    },
                )
                tool = outbound["messages"][-1]
                self.assertEqual(tool["role"], "tool")
                self.assertEqual(tool["tool_call_id"], "call_render")
                self.assertEqual(
                    tool["content"],
                    [
                        {"type": "text", "text": "Render complete"},
                        {
                            "type": "image_url",
                            "image_url": {"url": image, "detail": "high"},
                        },
                    ],
                )
                self.assertEqual(
                    outbound["messages"][-2]["tool_calls"][0]["id"], "call_render"
                )

    def test_tool_image_roundtrip(self):
        tool = {
            "role": "tool",
            "tool_call_id": "call_render",
            "content": [
                {"type": "text", "text": "Rendered"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,iVBORw0KGgo=",
                        "detail": "high",
                    },
                },
            ],
        }
        items = self.converter.chat_completions_messages_to_responses_items([tool])
        restored = self.converter.responses_to_chat_completion_create_params(
            NeMoGymResponseCreateParamsNonStreaming(input=items)
        )
        self.assertEqual(restored.model_dump()["messages"], [tool])

    def test_tool_image_unsupported_representations_rejected(self):
        for part in (
            {"type": "input_image", "file_id": "file_123"},
            {"type": "input_file", "file_id": "file_123"},
            {
                "type": "input_image",
                "image_url": "https://example.invalid/image.png",
                "detail": "original",
            },
        ):
            with self.subTest(part=part), self.assertRaises(NotImplementedError):
                self.converter.responses_to_chat_completion_create_params(
                    NeMoGymResponseCreateParamsNonStreaming(
                        input=[
                            {
                                "type": "function_call_output",
                                "call_id": "call_1",
                                "output": [part],
                            }
                        ]
                    )
                )

    def test_assistant_reasoning_preserved_after_tool(self):
        reasoning = "Check the chart before changing its layout."
        for stream in (False, True):
            with self.subTest(stream=stream):
                outbound = self.check_request(
                    "/v1/chat/completions",
                    {
                        "stream": stream,
                        "messages": [
                            {"role": "user", "content": "Inspect the chart."},
                            {
                                "role": "assistant",
                                "content": "",
                                "reasoning": reasoning,
                                "tool_calls": [
                                    {
                                        "type": "function",
                                        "id": "call_read",
                                        "function": {"name": "read", "arguments": "{}"},
                                    }
                                ],
                            },
                            {
                                "role": "tool",
                                "tool_call_id": "call_read",
                                "content": "Ready",
                            },
                        ],
                    },
                )
                self.assertEqual(outbound["messages"][1]["reasoning"], reasoning)

    def test_real_opencode_image_followup(self):
        if self.opencode_requests is None:
            self.skipTest(
                "Supply --opencode-requests from the actual read-image CLI probe"
            )
        followups = [
            body
            for body in self.opencode_requests
            if any(m["role"] == "tool" for m in body["messages"])
        ]
        self.assertTrue(followups, "Probe must include actual tool feedback")
        image_count = 0
        for body in followups:
            outbound = self.check_request("/v1/chat/completions", copy.deepcopy(body))
            for source, target in zip(
                body["messages"], outbound["messages"], strict=True
            ):
                self.assertEqual(source["role"], target["role"])
                if "reasoning" in source:
                    self.assertEqual(source["reasoning"], target["reasoning"])
                if isinstance(source.get("content"), list):
                    for part in source["content"]:
                        if part["type"] == "image_url":
                            image_count += 1
                            self.assertIn(part, target["content"])
        self.assertGreater(image_count, 0)

    def test_real_hermes_image_followup(self):
        if self.hermes_requests is None:
            self.skipTest("Supply --hermes-requests from the actual vision tool probe")
        image_count = 0
        reasoning_count = 0
        for body in self.hermes_requests:
            for stream in (False, True):
                outbound = self.check_request(
                    "/v1/chat/completions", copy.deepcopy(body) | {"stream": stream}
                )
                self.assertEqual(
                    outbound["chat_template_kwargs"], body["chat_template_kwargs"]
                )
                for source, target in zip(
                    body["messages"], outbound["messages"], strict=True
                ):
                    if isinstance(source.get("reasoning_content"), str):
                        reasoning_count += 1
                        self.assertEqual(
                            target["reasoning"], source["reasoning_content"]
                        )
                    if isinstance(source.get("content"), list):
                        for part in source["content"]:
                            if part["type"] == "image_url":
                                image_count += 1
                                self.assertIn(part, target["content"])
        self.assertGreater(image_count, 0)
        self.assertGreater(reasoning_count, 0)

    def test_conflicting_reasoning_rejected(self):
        response = self.client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {
                        "role": "assistant",
                        "content": "",
                        "reasoning": "A",
                        "reasoning_content": "B",
                    }
                ]
            },
        )
        self.assertEqual(response.status_code, 422)
        self.backend.create_chat_completion.assert_not_called()

    def test_conflicting_template_options_rejected(self):
        response = self.client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "chat_template_kwargs": {"enable_thinking": True},
                "metadata": {
                    "chat_template_kwargs": json.dumps({"enable_thinking": False})
                },
            },
        )
        self.assertEqual(response.status_code, 422)
        self.backend.create_chat_completion.assert_not_called()

    def test_real_codex_image_and_reasoning_followup(self):
        if self.codex_image_requests is None:
            self.skipTest(
                "Supply --codex-image-requests from the actual view_image CLI probe"
            )
        image_count = 0
        reasoning_count = 0
        for body in self.codex_image_requests:
            outbound = self.check_request("/v1/responses", copy.deepcopy(body))
            for item in body["input"]:
                if item.get("type") == "reasoning":
                    for part in item.get("summary") or []:
                        reasoning_count += 1
                        self.assertTrue(
                            any(
                                m["role"] == "assistant"
                                and part["text"] == m.get("reasoning")
                                for m in outbound["messages"]
                            ),
                            "Replayed thinking must reach the model",
                        )
                if item.get("type") == "function_call_output" and isinstance(
                    item.get("output"), list
                ):
                    for part in item["output"]:
                        if part["type"] == "input_image":
                            image_count += 1
                            tool = next(
                                m
                                for m in outbound["messages"]
                                if m.get("tool_call_id") == item["call_id"]
                            )
                            self.assertTrue(
                                any(
                                    p.get("image_url", {}).get("url")
                                    == part["image_url"]
                                    for p in tool["content"]
                                )
                            )
        self.assertGreater(image_count, 0)
        self.assertGreater(reasoning_count, 0)

    def test_assistant_message_replay_metadata(self):
        from nemo_gym.responses_streaming import sanitize_streaming_responses_body

        item = {
            "type": "message",
            "role": "assistant",
            "phase": "commentary",
            "content": [{"type": "output_text", "text": "Inspecting the render."}],
        }
        body = {"stream": True, "input": [item]}
        original = copy.deepcopy(body)
        cleaned, _ = sanitize_streaming_responses_body(body)
        repeated, _ = sanitize_streaming_responses_body(body)
        self.assertEqual(cleaned, repeated)
        self.assertEqual(body, original)
        self.assertTrue(cleaned["input"][0]["id"].startswith("msg_replay_"))
        self.assertEqual(cleaned["input"][0]["content"][0]["annotations"], [])
        outbound = self.check_request("/v1/responses", body)
        self.assertEqual(outbound["messages"][0]["content"], item["content"][0]["text"])
        self.assertEqual(outbound["messages"][0]["phase"], "commentary")

    def test_assistant_phase_boundaries(self):
        for phases in (("commentary", "final_answer"), ("commentary", None)):
            items = []
            for index, phase in enumerate(phases):
                item = {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": f"part{index}"}],
                }
                if phase is not None:
                    item["phase"] = phase
                items.append(item)
            outbound = self.check_request(
                "/v1/responses", {"stream": True, "input": items}
            )
            self.assertEqual(
                [m["content"] for m in outbound["messages"]], ["part0", "part1"]
            )
            self.assertEqual(
                [m.get("phase") for m in outbound["messages"]], list(phases)
            )

    def test_malformed_replayed_message_rejected(self):
        for role in ("assistant", "user", "developer", "system"):
            with self.subTest(role=role):
                self.backend.create_chat_completion.reset_mock()
                response = self.client.post(
                    "/v1/responses",
                    json={
                        "stream": True,
                        "input": [
                            {
                                "type": "message",
                                "role": role,
                                "content": [{"type": "output_text"}],
                            }
                        ],
                    },
                )
                self.assertEqual(response.status_code, 422)
                self.backend.create_chat_completion.assert_not_called()

    def test_real_codex_message_image_followup(self):
        if self.codex_message_requests is None:
            self.skipTest("Supply --codex-message-requests from the actual CLI probe")
        checked = 0
        for body in self.codex_message_requests:
            source_messages = [
                item
                for item in body["input"]
                if item.get("type") == "message" and item.get("role") == "assistant"
            ]
            if not source_messages:
                continue
            outbound = self.check_request("/v1/responses", copy.deepcopy(body))
            for source in source_messages:
                text = "".join(p.get("text", "") for p in source["content"])
                target = next(
                    m
                    for m in outbound["messages"]
                    if m["role"] == "assistant" and text in str(m.get("content"))
                )
                self.assertEqual(target.get("phase"), source.get("phase"))
                checked += 1
            self.assertTrue(
                any(
                    m["role"] == "tool"
                    and isinstance(m["content"], list)
                    and any(p.get("type") == "image_url" for p in m["content"])
                    for m in outbound["messages"]
                )
            )
        self.assertGreater(checked, 0)

    def test_malformed_replayed_reasoning_rejected(self):
        response = self.client.post(
            "/v1/responses",
            json={
                "stream": True,
                "input": [
                    {
                        "type": "reasoning",
                        "summary": [{"type": "summary_text", "text": 123}],
                    }
                ],
            },
        )
        self.assertEqual(response.status_code, 422)
        self.backend.create_chat_completion.assert_not_called()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codex-request", type=Path)
    parser.add_argument("--opencode-requests", type=Path)
    parser.add_argument("--hermes-requests", type=Path)
    parser.add_argument("--codex-image-requests", type=Path)
    parser.add_argument("--codex-message-requests", type=Path)
    args = parser.parse_args()
    if args.codex_request:
        ProtocolTests.wire_request = json.loads(args.codex_request.read_text())
    if args.opencode_requests:
        ProtocolTests.opencode_requests = json.loads(args.opencode_requests.read_text())
    if args.hermes_requests:
        ProtocolTests.hermes_requests = json.loads(args.hermes_requests.read_text())
    if args.codex_image_requests:
        ProtocolTests.codex_image_requests = json.loads(
            args.codex_image_requests.read_text()
        )
    if args.codex_message_requests:
        ProtocolTests.codex_message_requests = json.loads(
            args.codex_message_requests.read_text()
        )
    unittest.main(argv=[__file__], verbosity=2)
