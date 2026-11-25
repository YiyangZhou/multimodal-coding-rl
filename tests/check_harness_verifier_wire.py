# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Preserve trusted verifier extensions across the native HarnessAgent /run route.

Uses the actual FastAPI handler and response serialization with synthetic
model/resource services. No real model, remote sandbox or GPU is used.
"""

import json
import sys
import unittest
from contextlib import redirect_stdout
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient
from mmrl.process_rewards import ProcessRewardConfig
from nemo_gym import PARENT_DIR
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from responses_api_agents.harness_agent import app as native


def through_harness(verifier_fields: dict, *, agent_name: str = "codex") -> dict:
    """Run a trusted synthetic verifier result through the real /run boundary."""
    if not Path(native.__file__).resolve().is_relative_to(PARENT_DIR.resolve()):
        raise RuntimeError(
            "HarnessAgent came from another Gym checkout; set NEMO_GYM_EXTRA_ROOTS to the tested Gym"
        )
    final = NeMoGymResponse(
        id="resp-wire",
        created_at=1,
        model="policy",
        object="response",
        status="completed",
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
        output=[
            {
                "type": "message",
                "id": "msg-wire",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {"type": "output_text", "text": "<svg/>", "annotations": []}
                ],
            }
        ],
    )
    body = {
        "responses_create_params": {"model": "policy", "input": "Draw a circle"},
        "_ng_rollout_id": "wire-rollout",
        "task": {"task_id": "wire-case"},
        "route": agent_name,
    }
    verified = {**body, "response": final.model_dump(mode="json"), **verifier_fields}

    def response(payload: dict) -> SimpleNamespace:
        return SimpleNamespace(
            ok=True,
            cookies={},
            read=AsyncMock(return_value=json.dumps(payload).encode()),
        )

    client = MagicMock(spec=ServerClient, global_config_dict={})
    client.post = AsyncMock(
        side_effect=[
            response({"sandbox_descriptor": {"sandbox_id": "synthetic"}}),
            response(verified),
        ]
    )
    config = native.HarnessAgentConfig(
        host="127.0.0.1",
        port=18888,
        entrypoint="app.py",
        name="wire-test",
        resources_server={"type": "resources_servers", "name": "resource"},
        model_server={"type": "responses_api_models", "name": "policy_model"},
        agent=agent_name,
        sandbox_provider="synthetic",
        gym_source="/tmp/unused-gym-source.tar",
    )
    with (
        patch.object(native, "resolve_provider_config", return_value={}),
        patch.object(native, "resolve_provider_metadata", return_value={}),
        patch.object(native, "create_provider", return_value=MagicMock()),
        patch.object(native.HarnessAgent, "responses", AsyncMock(return_value=final)),
    ):
        agent = native.HarnessAgent(config=config, server_client=client)
        with TestClient(agent.setup_webserver()) as http:
            result = http.post("/run", json=body)
        if result.status_code != 200:
            raise AssertionError(f"Native /run failed: {result.status_code}")
    assert client.post.await_count == 2
    for call in client.post.await_args_list:
        assert call.kwargs["json"]["_ng_rollout_id"] == body["_ng_rollout_id"]
    return result.json()


class VerifierWireTests(unittest.TestCase):
    def test_harnesses_preserve_measured_rewards_and_infrastructure_exclusion(self):
        for agent in ("opencode", "codex", "hermes"):
            for status in ("success", "execution_failure", "infrastructure_failure"):
                with self.subTest(agent=agent, status=status):
                    excluded = status == "infrastructure_failure"
                    score = 0.8 if status == "success" else 0.0
                    fields = {
                        "reward": score,
                        "mask_sample": excluded,
                        "instance_config": {"mask_sample": excluded},
                        "terminal_response_id": "resp-wire",
                    }
                    if status != "success":
                        fields["failure_kind"] = "mm_coding:" + (
                            "infrastructure" if excluded else "execution_failed"
                        )
                        fields["failure_reason"] = "synthetic boundary regression"
                    if not excluded:
                        fields["process_reward_payload"] = {
                            "version": 1,
                            "rollout_id": "wire-rollout",
                            "final_reward": score,
                            "stages": [],
                            "config": asdict(ProcessRewardConfig()),
                        }
                    wire = through_harness(fields, agent_name=agent)
                    for key, value in fields.items():
                        self.assertEqual(wire.get(key), value, key)
                    self.assertEqual("process_reward_payload" in wire, not excluded)


if __name__ == "__main__":
    if sys.argv[1:] == ["--transport-stream"]:
        for line in sys.stdin:
            with redirect_stdout(sys.stderr):
                result = through_harness(json.loads(line))
            print(json.dumps(result), flush=True)
    else:
        unittest.main(verbosity=2)
