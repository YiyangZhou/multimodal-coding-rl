"""Carry a real process deadline through OpenCode's response and Gym masking."""
import asyncio
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from omegaconf import OmegaConf
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import BaseServerConfig, ServerClient
from responses_api_agents.harness_agent.app import HarnessAgentVerifyResponse
from mmrl.gym_app import CodingResourcesServer, Config
from mmrl.tasks import Task, gym_row


async def main():
    # Candidate testing can select an isolated native source without modifying
    # a running Gym installation. Installed checks use the normal module.
    override = os.environ.get("MMRL_OPENCODE_SOURCE")
    if override:
        name = "responses_api_agents.opencode_agent.app"
        spec = importlib.util.spec_from_file_location(name, override)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        import responses_api_agents.opencode_agent as package
        package.app = module
    from responses_api_agents.opencode_agent import app as native

    # Use an actual OS process so communicate cancellation, kill and reap are
    # exercised. No model endpoint, external CLI or token budget is mocked as
    # having completed: this checks the adapter's wall-deadline path only.
    spawn = asyncio.create_subprocess_exec
    processes = []
    async def sleeping_process(*args, **kwargs):
        proc = await spawn(sys.executable, "-c", "import time; time.sleep(30)",
                           stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        processes.append(proc)
        return proc

    with tempfile.TemporaryDirectory() as directory:
        cfg = native.OpenCodeAgentConfig(
            host="127.0.0.1", port=18001, name="deadline-test", entrypoint="app.py",
            resources_server={"type": "resources_servers", "name": "unused"},
            workspace_root=directory, command="opencode", timeout=1,
        )
        with patch.object(native, "ensure_opencode"):
            agent = native.OpenCodeAgent(config=cfg, server_client=MagicMock(spec=ServerClient))
        with patch("asyncio.create_subprocess_exec", side_effect=sleeping_process):
            response = await agent.responses(SimpleNamespace(path_params={}),
                NeMoGymResponseCreateParamsNonStreaming(input="Deadline control"))
    assert len(processes) == 1 and processes[0].returncode is not None
    assert response.status == "incomplete"
    assert (response.metadata or {}).get("interrupted") == "true", "native timeout lost interruption marker"

    client = ServerClient(head_server_config=BaseServerConfig(host="127.0.0.1", port=18003),
                          global_config_dict=OmegaConf.create({}))
    with patch("mmrl.gym_app.RemoteSandbox"), patch("mmrl.gym_app.QualityJudge"):
        server = CodingResourcesServer(config=Config(
            host="127.0.0.1", port=18004, name="mm_coding", entrypoint="app.py",
            sandbox_image="registry.invalid/img@sha256:" + "a" * 64), server_client=client)
    server._sandbox = AsyncMock()
    server._sandbox.create.return_value = object()
    server._sandbox.provider.serialize_handle.return_value = {"id": "synthetic"}
    server._service.score = AsyncMock()
    row = gym_row(Task("mmsvg_text2svg", "deadline", "Draw a circle"), "opencode")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.setup_webserver()),
                                base_url="http://test") as http:
        assert (await http.post("/seed_session", json=row)).status_code == 200
        result = await http.post("/verify", json=row | {"response": response.model_dump(mode="json")})
    assert result.status_code == 200, result.text
    body = HarnessAgentVerifyResponse.model_validate(result.json()).model_dump(mode="json")
    assert body["mask_sample"] and body["instance_config"]["mask_sample"]
    assert body["failure_kind"] == "mm_coding:infrastructure"
    server._service.score.assert_not_awaited()
    server._sandbox.read_artifact.assert_not_awaited()
    server._sandbox.provider.close.assert_awaited_once()
    return {"passed": True, "real_subprocess_deadline": True,
            "native_unscoped_response": True, "native_http_masked": True,
            "judge_calls": 0, "real_opencode_cli": False,
            "source_sha256": __import__('hashlib').sha256(Path(native.__file__).read_bytes()).hexdigest()}


if __name__ == "__main__":
    print(json.dumps(asyncio.run(main())))
