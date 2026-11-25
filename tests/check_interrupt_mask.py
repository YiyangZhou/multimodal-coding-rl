"""Check interruption versus model incompletion through native verification HTTP."""
import asyncio
from unittest.mock import AsyncMock, patch

import httpx
from omegaconf import OmegaConf
from nemo_gym.server_utils import BaseServerConfig, ServerClient
from responses_api_agents.harness_agent.app import HarnessAgentVerifyResponse
from responses_api_agents.simple_agent.app import SimpleAgentVerifyResponse

from mmrl.gym_app import CodingResourcesServer, Config
from mmrl.tasks import Task, gym_row


async def main():
    client = ServerClient(
        head_server_config=BaseServerConfig(host="127.0.0.1", port=18003),
        global_config_dict=OmegaConf.create({}),
    )
    with patch("mmrl.gym_app.RemoteSandbox"), patch("mmrl.gym_app.QualityJudge"):
        server = CodingResourcesServer(
            config=Config(host="127.0.0.1", port=18004, name="mm_coding",
                          entrypoint="app.py", sandbox_image="registry.invalid/img@sha256:" + "a" * 64),
            server_client=client,
        )
    server._service.score = AsyncMock(return_value={"reward": 0.76, "execution": 1.0})
    row = gym_row(Task("mmsvg_text2svg", "interrupt-control", "Draw a circle"), "single")
    response = {
        "id": "resp-control", "created_at": 1.0, "model": "policy", "object": "response",
        "tool_choice": "auto", "tools": [], "parallel_tool_calls": True,
        "output": [{"id": "msg-control", "type": "message", "role": "assistant", "status": "completed",
                    "content": [{"type": "output_text", "text": "<svg/>", "annotations": []}]}],
    }
    cases = []
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.setup_webserver()),
                                base_url="http://test") as http:
        for status, interrupted, excluded, scored in (
            ("incomplete", "true", True, False),
            ("completed", "true", True, False),
            ("incomplete", "false", False, False),
            ("completed", "false", False, True),
        ):
            server._service.score.reset_mock()
            seeded = await http.post("/seed_session", json=row)
            assert seeded.status_code == 200
            result = await http.post("/verify", json=row | {"response": response | {
                "status": status, "metadata": {"interrupted": interrupted}}})
            assert result.status_code == 200, result.text
            body = result.json()
            assert body["mask_sample"] is excluded, (status, interrupted, body["failure_kind"])
            assert bool(server._service.score.await_count) is scored
            if excluded:
                assert body["failure_kind"] == "mm_coding:infrastructure"
            elif not scored:
                assert body["failure_kind"] == "mm_coding:incomplete" and body["reward"] == 0
            else:
                assert body["reward"] == 0.76
            for wire_type in (HarnessAgentVerifyResponse, SimpleAgentVerifyResponse):
                wire = wire_type.model_validate(body).model_dump(mode="json")
                assert wire["mask_sample"] is excluded
                assert wire["instance_config"]["mask_sample"] is excluded
            assert not server._sessions
            cases.append({"status": status, "interrupted": interrupted, "excluded": excluded, "scored": scored})
    return {"passed": True, "native_http_cases": cases, "native_agent_wire_types": 2,
            "real_model_acceptance": False}


if __name__ == "__main__":
    import json
    print(json.dumps(asyncio.run(main())))
