# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Remote runner failures reach the shared verifier without a model zero score."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from mmrl.gym_app import CodingResourcesServer, Config
from mmrl.tasks import Task, gym_row
from omegaconf import OmegaConf
from responses_api_agents.harness_agent.app import HarnessAgent, HarnessAgentVerifyResponse
from responses_api_agents.harness_agent.tests.test_app import _make_agent, _response

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox.providers.base import SandboxExecResult
from nemo_gym.server_utils import BaseServerConfig, ServerClient


async def invoke(harness, code, marker, enabled=True):
    agent = _make_agent(agent=harness, runner_failure_as_response=enabled)
    handle = MagicMock(provider_name="opensandbox")
    agent._provision_box = AsyncMock(return_value=handle)
    agent._provider.exec = AsyncMock(
        side_effect=[
            SandboxExecResult("", "private stderr should not enter response", code),
            SandboxExecResult(marker, "", 0),
        ]
    )
    agent._provider.close = AsyncMock()
    body = NeMoGymResponseCreateParamsNonStreaming(input="hello")
    with (
        patch.object(agent, "_sandbox_model_url", return_value="https://model.invalid"),
        patch.object(HarnessAgent, "_download_json", new=AsyncMock(return_value=_response())),
    ):
        response = await agent.responses(MagicMock(), body)
    agent._provider.close.assert_awaited_once_with(handle)
    return response


async def verify_failed(response, route):
    client = ServerClient(
        head_server_config=BaseServerConfig(host="127.0.0.1", port=18003), global_config_dict=OmegaConf.create({})
    )
    with patch("mmrl.gym_app.RemoteSandbox"), patch("mmrl.gym_app.QualityJudge"):
        server = CodingResourcesServer(
            config=Config(
                host="127.0.0.1",
                port=18004,
                name="mm_coding",
                entrypoint="app.py",
                sandbox_image="registry.invalid/img@sha256:" + "a" * 64,
            ),
            server_client=client,
        )
    handle = MagicMock(name="resource-owned-handle")
    server._sandbox.create = AsyncMock(return_value=handle)
    server._sandbox.prepare = AsyncMock()
    server._sandbox.provider.serialize_handle = AsyncMock(return_value={"sandbox_id": "fake-owned-box"})
    server._sandbox.provider.close = AsyncMock()
    server._service.score = AsyncMock()
    row = gym_row(Task("mmsvg_text2svg", "runner-failure", "Draw a circle"), route)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.setup_webserver()), base_url="http://test"
    ) as http:
        assert (await http.post("/seed_session", json=row)).status_code == 200
        result = await http.post("/verify", json=row | {"response": response.model_dump(mode="json")})
    assert result.status_code == 200
    wire = HarnessAgentVerifyResponse.model_validate(result.json()).model_dump(mode="json")
    assert wire["mask_sample"] is True and wire["instance_config"]["mask_sample"] is True
    assert wire["failure_kind"] == "mm_coding:infrastructure"
    assert "execution" not in wire
    server._service.score.assert_not_awaited()
    assert not server._sessions
    server._sandbox.provider.close.assert_awaited_once_with(handle)


@pytest.mark.parametrize("harness", ["opencode", "codex", "hermes"])
@pytest.mark.parametrize("code,marker", [(-1, ""), (1, "RUNNER_DONE"), (0, "missing marker")])
def test_runner_failure_is_masked_without_scoring(harness, code, marker):
    async def run():
        response = await invoke(harness, code, marker)
        assert response.status == "failed" and response.error.message == "harness_runner_failed"
        assert response.output == []
        assert "private stderr" not in response.model_dump_json()
        await verify_failed(response, harness)

    asyncio.run(run())


def test_default_retains_native_error_behavior():
    with pytest.raises(RuntimeError, match="runner failed"):
        asyncio.run(invoke("hermes", -1, "", enabled=False))


def test_completed_runner_preserves_response():
    response = asyncio.run(invoke("hermes", 0, "RUNNER_DONE"))
    assert response.model_dump(exclude_none=True)["id"] == "response-1"
    assert response.error is None and response.status != "failed"
