"""Actual HarnessAgent HTTP -> resource session -> per-rollout audit identity.

Harness execution and the quality service are mocked; this is not GPU acceptance.
"""
import asyncio
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from omegaconf import OmegaConf
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient, BaseServerConfig
from responses_api_agents.harness_agent.app import HarnessAgent, HarnessAgentConfig
from responses_api_agents.simple_agent.app import SimpleAgent, SimpleAgentConfig
from mmrl.gym_app import CodingResourcesServer, Config
from mmrl.tasks import Task, gym_row


async def main():
    sc = ServerClient(
        head_server_config=BaseServerConfig(host="127.0.0.1", port=18003),
        global_config_dict=OmegaConf.create({}),
    )
    with patch("mmrl.gym_app.RemoteSandbox"), patch("mmrl.gym_app.QualityJudge"):
        resource = CodingResourcesServer(config=Config(
            host="127.0.0.1", port=18004, name="mm_coding", entrypoint="app.py",
            sandbox_image="example.invalid/test@sha256:" + "a"*64,
        ), server_client=sc)
    resource._service.score = AsyncMock(return_value={"reward": 0.76, "execution": 1.0})
    resource._sandbox.create = AsyncMock(return_value="owned-test-box")
    resource._sandbox.prepare = AsyncMock()
    resource._sandbox.read_artifact = AsyncMock(return_value="<svg/>")
    resource._sandbox.provider.serialize_handle = AsyncMock(return_value={"sandbox_id": "owned-test-box"})
    resource._sandbox.provider.close = AsyncMock()
    response = NeMoGymResponse(
        id="shared-response-id", created_at=1.0, model="policy", object="response",
        status="completed", parallel_tool_calls=True, tool_choice="auto", tools=[],
        output=[{"type":"message", "id":"msg-test", "role":"assistant", "status":"completed",
                 "content":[{"type":"output_text","text":"<svg/>","annotations":[]}]}],
    )
    observed = []
    expected = {}
    with tempfile.TemporaryDirectory() as directory:
        resource.config.verification_log_dir = directory
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=resource.setup_webserver()), base_url="http://resources") as resource_http:
            async def post(*, server_name, url_path, json, cookies):
                if url_path.endswith("/v1/responses"):
                    # The single-turn agent's model endpoint is mocked, while
                    # /run, resource session and verification use real ASGI.
                    return SimpleNamespace(cookies=cookies, http_response=httpx.Response(
                        200, json=returned.model_dump(mode="json"), request=httpx.Request("POST", "http://model/v1/responses")))
                observed.append((url_path, json.get("_ng_rollout_id")))
                response = await resource_http.post(url_path, json=json, cookies=cookies)
                return SimpleNamespace(cookies=dict(response.cookies), http_response=response)
            async def read_json(result):
                return result.http_response.json()
            async def check_status(result):
                result.http_response.raise_for_status()
            for route in ("single", "opencode", "codex", "hermes"):
                client = MagicMock(spec=ServerClient)
                client.global_config_dict = {}
                client.post = AsyncMock(side_effect=post)
                config_type = SimpleAgentConfig if route == "single" else HarnessAgentConfig
                config = config_type(
                    host="127.0.0.1", port=18000, name="identity_"+route, entrypoint="",
                    resources_server=ResourcesServerRef(type="resources_servers", name="rs"),
                    model_server=ModelServerRef(type="responses_api_models", name="model"),
                    agent=route, sandbox_provider={"opensandbox": {}},
                )
                with patch("responses_api_agents.harness_agent.app.create_provider"), patch.object(HarnessAgent,"_build_gym_tar",return_value=None):
                    agent_type = SimpleAgent if route == "single" else HarnessAgent
                    agent = agent_type(config=config, server_client=client)
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=agent.setup_webserver()),base_url="http://agent") as agent_http:
                    for generation, masked in enumerate((False, True)):
                        identity=f"group-{route}_g{generation}_aattempt"
                        row=gym_row(Task("mmsvg_text2svg","same-task","Draw a circle"),route) | {"_ng_rollout_id":identity}
                        returned=response.model_copy(deep=True)
                        if masked:
                            returned.status="failed"
                        with patch.object(HarnessAgent,"responses",new=AsyncMock(return_value=returned)), patch("responses_api_agents.harness_agent.app.raise_for_status",new=check_status), patch("responses_api_agents.harness_agent.app.get_response_json",new=read_json), patch("responses_api_agents.simple_agent.app.raise_for_status",new=check_status), patch("responses_api_agents.simple_agent.app.get_response_json",new=read_json):
                            reply=await agent_http.post("/run",json=row)
                        assert reply.status_code==200, reply.text
                        assert reply.json()["mask_sample"] is masked
                        assert "_ng_rollout_id" not in reply.json(), "Internal identity should remain excluded from public model schema"
                        expected[identity]=masked
                        assert not resource._sessions
            records=[json.loads(p.read_text()) for p in Path(directory).glob("*/record.json")]
            assert len(records)==8 and len({r["rollout_id"] for r in records})==8
            assert all(r["complete"] and bool(r["result"].get("mask_sample")) == expected[r["rollout_id"]] for r in records)
            # Deliberately identical response IDs cannot accidentally provide the join.
            assert {r["response_id"] for r in records} == {"shared-response-id"}
            assert sorted(observed)==sorted((endpoint, identity) for identity in expected for endpoint in ("/seed_session","/verify"))
            assert resource._service.score.await_count==4
            assert resource._sandbox.provider.close.await_count==6
            # Changing the identity after seed must fail, never score another row.
            row=gym_row(Task("mmsvg_text2svg","mismatch","Draw a circle"),"single") | {"_ng_rollout_id":"seed-id"}
            assert (await resource_http.post("/seed_session",json=row)).status_code==200
            before=resource._service.score.await_count
            try:
                await resource_http.post("/verify",json=row | {"_ng_rollout_id":"different-id","response":response.model_dump(mode="json")})
            except RuntimeError as error:
                assert str(error)=="Rollout identity changed during verification"
            else:
                raise AssertionError("Identity mismatch was accepted")
            assert resource._service.score.await_count==before and not resource._sessions
    print(json.dumps({"passed":True,"actual_harness_http_routes":3,"single_turn_http":True,"matched_records":8,"mixed_failure_rows":4,"identity_mismatch_rejected":True,"GPU_or_real_harness_execution":False}))


if __name__ == "__main__":
    asyncio.run(main())
