"""Actual TLS lifecycle with Gym model address resolution; no model inference."""

import asyncio
import importlib.util
import os
from pathlib import Path
import socket
import ssl
import tempfile
from unittest.mock import patch

import aiohttp
from aiohttp import web
from omegaconf import OmegaConf
from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig

from mmrl.config import gym_config
from mmrl.tls import managed_gateway


async def main():
    repo = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "create_ca", repo / "scripts/create_capture_ca.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []
    observations = []

    async def observe(*args):
        observations.append(args)

    async def reply(request):
        calls.append((request.path, await request.read()))
        return web.Response(
            body=b'data: {"captured":true}\n\ndata: [DONE]\n\n',
            content_type="text/event-stream",
        )

    backend = web.Application()
    backend.router.add_post("/{path:.*}", reply)
    runner = web.AppRunner(backend)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    backend_port = site._server.sockets[0].getsockname()[1]
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        gateway_port = sock.getsockname()[1]
    cfg = OmegaConf.create(gym_config("image@sha256:" + "a" * 64, "auto"))
    cfg.default_host = "127.0.0.1"
    discovery = OmegaConf.merge(
        GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT, cfg
    )
    with patch.dict(
        os.environ,
        {
            "OPENSANDBOX_DOMAIN": "sandbox.example.invalid",
            "OPENSANDBOX_API_KEY": "test-only",
            "MMRL_CAPTURE_GATEWAY_TOKEN": "test-only",
        },
    ):
        parser = GlobalConfigDictParser()
        resolved = parser.parse(
            GlobalConfigDictParserConfig(
                initial_global_config_dict=discovery,
                skip_load_from_cli=True,
                skip_load_from_dotenv=True,
                offline=True,
            )
        )
        instances = parser.filter_for_server_instance_configs(resolved)
        assert {server.name for server in instances} == {
            "policy_model",
            "mm_coding",
            "mm_single",
            "mm_opencode",
            "mm_codex",
            "mm_hermes",
        }
    assert cfg.mm_coding.resources_servers.mm_coding.host == "127.0.0.1"
    for harness in ("opencode", "codex", "hermes"):
        assert (
            cfg[
                f"mm_{harness}"
            ].responses_api_agents.harness_agent.sandbox_model_base_url
            == "https://127.0.0.1:5999"
        )
    cfg.policy_model = {
        "responses_api_models": {
            "vllm_model": {"host": "127.0.0.1", "port": backend_port}
        }
    }
    try:
        with tempfile.TemporaryDirectory() as temporary:
            ca = Path(temporary) / "ca"
            module.create(ca)
            env = {
                "MMRL_CAPTURE_CA_CERT": str(ca / "ca.pem"),
                "MMRL_CAPTURE_CA_KEY": str(ca / "ca.key"),
                "MMRL_CAPTURE_GATEWAY_TOKEN": "test-" * 12,
            }
            with patch.dict(os.environ, env):
                async with managed_gateway(gateway_port, "127.0.0.1", cfg,
                                           response_observer=observe,
                                           tool_upstream=f"http://127.0.0.1:{backend_port}"):
                    context = ssl.create_default_context(cafile=str(ca / "ca.pem"))
                    url = f"https://127.0.0.1:{gateway_port}/ng-rollout/real-lifecycle-test/training-token-capture/v1/chat/completions"
                    async with aiohttp.ClientSession(
                        connector=aiohttp.TCPConnector(ssl=context)
                    ) as client:
                        async with client.post(url, data=b"{}") as response:
                            assert response.status == 401
                        async with client.post(
                            url,
                            data=b'{"input":"image-turn"}',
                            headers={
                                "Authorization": "Bearer "
                                + env["MMRL_CAPTURE_GATEWAY_TOKEN"]
                            },
                        ) as response:
                            assert response.status == 200
                            assert (
                                await response.read()
                                == b'data: {"captured":true}\n\ndata: [DONE]\n\n'
                            )
                        async with client.post(
                            f"https://127.0.0.1:{gateway_port}/mcp", data=b'{"method":"tools/list"}',
                            headers={"Authorization": "Bearer " + env["MMRL_CAPTURE_GATEWAY_TOKEN"]},
                        ) as response:
                            assert response.status == 200
                            await response.read()
                    assert calls == [
                        (
                            "/ng-rollout/real-lifecycle-test/training-token-capture/v1/chat/completions",
                            b'{"input":"image-turn"}',
                        ),
                        ("/mcp", b'{"method":"tools/list"}'),
                    ]
                    assert len(observations) == 1
                    assert observations[0][0] == calls[0][0]
                    async with aiohttp.ClientSession() as untrusted:
                        try:
                            await untrusted.post(url)
                        except aiohttp.ClientConnectorCertificateError:
                            pass
                        else:
                            raise AssertionError(
                                "An untrusted client accepted the private CA"
                            )
                with socket.socket() as sock:
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    sock.bind(("127.0.0.1", gateway_port))
    finally:
        await runner.cleanup()
    print(
        "Managed Gym gateway: address resolution, TLS verification, auth, model/MCP routing, observer, body/SSE and cleanup PASS"
    )


if __name__ == "__main__":
    asyncio.run(main())
