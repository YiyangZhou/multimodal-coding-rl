"""Run in the pinned ARM container. Exercises real Gym types, not a GPU update."""

import asyncio
import importlib
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

from mmrl.config import gym_config, training_config


def main():
    from nemo_gym.cli.env import _resolve_server_dir
    from nemo_gym.sandbox.providers.base import SandboxSpec
    from nemo_gym.token_id_capture.config import TokenIdCaptureConfig
    from responses_api_agents.harness_agent.app import HarnessAgentConfig, resolve_agent
    from responses_api_agents.simple_agent.app import SimpleAgentConfig
    from mmrl.gym_app import CodingResourcesServer, Config

    image = "example.invalid/mm-coding@sha256:" + "a" * 64
    cfg = gym_config(image, "https://capture.example.invalid")
    # Gym constructs its provider from a resolved dict, without dataclass type
    # coercion. Environment interpolation must preserve a boolean for aiohttp.
    from omegaconf import OmegaConf
    from nemo_gym.sandbox.providers.registry import create_provider

    for value, expected in ((None, True), ("true", True), ("false", False)):
        with patch.dict(
            os.environ,
            {
                "OPENSANDBOX_DOMAIN": "https://sandbox.example.invalid",
                "OPENSANDBOX_API_KEY": "test-placeholder",
            },
        ):
            if value is None:
                os.environ.pop("OPENSANDBOX_TLS_VERIFY", None)
            else:
                os.environ["OPENSANDBOX_TLS_VERIFY"] = value
            resolved = OmegaConf.to_container(
                OmegaConf.create(cfg["sandbox"]), resolve=True
            )
            provider = create_provider(resolved)
            assert provider._connection.tls_verify is expected

    for name in ("opencode", "codex", "hermes"):
        entry = cfg[f"mm_{name}"]["responses_api_agents"]["harness_agent"]
        outer = HarnessAgentConfig(
            host="127.0.0.1", port=18000, name=f"mm_{name}", **entry
        )
        module, _, config_name = resolve_agent(name)
        native = importlib.import_module(module)
        inner_cls = getattr(native, config_name)
        kwargs = dict(outer.agent_kwargs)
        kwargs.setdefault("resources_server", entry["resources_server"])
        kwargs.setdefault("model_server", entry["model_server"])
        inner = inner_cls(
            host="127.0.0.1",
            port=18001,
            name=name,
            entrypoint="app.py",
            **kwargs,
        )
        from mmrl.catalog import BENCHMARKS
        from mmrl.tasks import Task, gym_row
        from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming

        for benchmark, contract in BENCHMARKS.items():
            if name not in contract.routes:
                continue
            task = Task(
                benchmark,
                "projection-probe",
                "Preserve exact task α + β.",
                []
                if benchmark in ("mmsvg_text2svg", "webmmu_code_edit")
                else ["reference.png"],
                original_source="original" if benchmark == "webmmu_code_edit" else "",
                reference_patch="private reference diff"
                if benchmark == "webmmu_code_edit"
                else "",
            )
            row = gym_row(task, name)
            body = NeMoGymResponseCreateParamsNonStreaming.model_validate(
                row["responses_create_params"]
            )
            split = (
                native._split_input_to_user_and_history
                if name == "hermes"
                else native._extract_instruction
            )
            projected = split(body.input)
            assert task.prompt in projected[0]
            assert "/workspace/" + contract.artifact in projected[-1]
            assert "python /opt/mmrl/render.py" in projected[-1]
            assert (
                task.reference_patch not in str(projected)
                if task.reference_patch
                else True
            )
        if name == "opencode":
            # Exercise the native methods without invoking npm/CLI installation.
            with patch.object(native, "ensure_opencode"):
                agent = native.OpenCodeAgent.model_construct(
                    config=inner, server_client=None
                )
            effective = agent._build_opencode_config()
            policy = effective["provider"]["gym"]["models"]["policy"]
            assert policy["modalities"]["input"] == ["text", "image"]
            assert policy["interleaved"] == {"field": "reasoning"}
            assert effective["compaction"] == {"auto": False, "prune": False}
            assert agent._env("/tmp/mmrl-opencode-probe")[
                "OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX"
            ] == str(inner.max_output_tokens)
            assert agent._effective_model() == "gym/policy"
            assert set(effective["provider"]) == {"gym"}
            assert (
                effective["provider"]["gym"]["options"]["apiKey"]
                == "${oc.env:MMRL_CAPTURE_GATEWAY_TOKEN}"
            )
            for deadline in ("timeout", "headerTimeout", "chunkTimeout"):
                assert (
                    effective["provider"]["gym"]["options"][deadline]
                    == inner.timeout * 1000
                )
            assert (
                agent._env("/tmp/mmrl-opencode-probe")["OPENAI_API_KEY"]
                == "${oc.env:MMRL_CAPTURE_GATEWAY_TOKEN}"
            )
        if name == "hermes":
            assert inner.enabled_toolsets == ["terminal", "file", "vision"]
            assert inner.compression_enabled is False
            assert outer.sandbox_python.endswith("-m mmrl.hermes_runner")
        print("native harness config:", name, flush=True)
    SimpleAgentConfig(
        host="127.0.0.1",
        port=18002,
        name="mm_single",
        **cfg["mm_single"]["responses_api_agents"]["simple_agent"],
    )
    capture = TokenIdCaptureConfig(
        token_id_capture=dict(
            enabled=True,
            all_agents=True,
            rebuild_response=False,
            max_mask_fraction=0.05,
            dir=str(Path(tempfile.gettempdir()) / "mmrl-native-capture"),
        )
    )
    assert capture.token_id_capture.enabled
    SandboxSpec(image=image, resources={"cpu": 4, "memory_mib": 8192, "disk_gib": 20})
    server_dir = _resolve_server_dir(Path("resources_servers/mm_coding"))
    assert (
        server_dir.resolve()
        == Path(os.environ["MMRL_REPO"], "integrations/nemo_gym/resources_servers/mm_coding").resolve()
    )
    from nemo_gym.server_utils import ServerClient, BaseServerConfig

    sc = ServerClient(
        head_server_config=BaseServerConfig(host="127.0.0.1", port=18003),
        global_config_dict=OmegaConf.create({}),
    )
    with patch("mmrl.gym_app.RemoteSandbox"), patch("mmrl.gym_app.QualityJudge"):
        server = CodingResourcesServer(
            config=Config(
                host="127.0.0.1",
                port=18004,
                name="mm_coding",
                entrypoint="app.py",
                sandbox_image=image,
            ),
            server_client=sc,
        )
    app = server.setup_webserver()
    schema = app.openapi()
    assert {"/seed_session", "/verify"} <= schema["paths"].keys()
    # Real ASGI request parsing, cookies, resource lifecycle and completed-answer handling.
    import httpx
    from mmrl.tasks import Task, gym_row

    server._service.score = AsyncMock(return_value={"reward": 0.76, "execution": 1.0})

    async def roundtrip():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            row = gym_row(Task("mmsvg_text2svg", "case", "Draw a circle"), "single")
            seed = await client.post("/seed_session", json=row)
            assert seed.status_code == 200, seed.text
            from nemo_gym.openai_utils import NeMoGymResponse

            # Required fields are derived from the installed Responses schema.
            response = NeMoGymResponse(
                id="resp_test",
                created_at=1.0,
                model="policy",
                object="response",
                status="completed",
                output=[
                    {
                        "type": "message",
                        "id": "msg_test",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {"type": "output_text", "text": "<svg/>", "annotations": []}
                        ],
                    }
                ],
                tool_choice="auto",
                tools=[],
                parallel_tool_calls=True,
            )
            verified = await client.post(
                "/verify", json=row | {"response": response.model_dump(mode="json")}
            )
            assert verified.status_code == 200, verified.text
            assert verified.json()["reward"] == 0.76
            assert not server._sessions

            # Harness/provider failures must never become valid zero-reward examples.
            for failure in (
                {"status": "failed"},
                {"metadata": {"failed": "true"}},
                {"error": {"code": "server_error", "message": "provider unavailable"}},
                {"status": "failed", "error": {"code": "server_error", "message": "codex_nonzero_exit"}},
            ):
                server._service.score.reset_mock()
                await client.post("/seed_session", json=row)
                result = await client.post(
                    "/verify",
                    json=row | {"response": response.model_dump(mode="json") | failure},
                )
                assert result.status_code == 200, result.text
                assert result.json()["mask_sample"] is True
                assert result.json()["instance_config"]["mask_sample"] is True
                assert result.json()["failure_kind"] == "mm_coding:infrastructure"
                server._service.score.assert_not_awaited()
                assert not server._sessions

            from mmrl.reward import InfrastructureError

            server._service.score.side_effect = InfrastructureError("judge unavailable")
            await client.post("/seed_session", json=row)
            failed = await client.post(
                "/verify", json=row | {"response": response.model_dump(mode="json")}
            )
            assert failed.status_code == 200, failed.text
            assert failed.json()["mask_sample"] is True
            assert failed.json()["instance_config"]["mask_sample"] is True
            # Both native agents deserialize and serialize the verifier result.
            from responses_api_agents.harness_agent.app import (
                HarnessAgentVerifyResponse,
            )
            from responses_api_agents.simple_agent.app import SimpleAgentVerifyResponse
            from mmrl.gym_app import VerifyResponse

            for wire_type in (HarnessAgentVerifyResponse, SimpleAgentVerifyResponse):
                wire = wire_type.model_validate(failed.json()).model_dump(mode="json")
                assert wire["mask_sample"] is True
                assert wire["instance_config"]["mask_sample"] is True
                # A stale nested flag must never override the verifier's canonical flag.
                recovered = VerifyResponse.model_validate(
                    wire
                    | {
                        "mask_sample": False,
                        "instance_config": {"mask_sample": True, "tag": "kept"},
                    }
                ).model_dump(mode="json")
                assert recovered["instance_config"] == {
                    "mask_sample": False,
                    "tag": "kept",
                }
            assert not server._sessions
            server._service.score.side_effect = None

            server._sandbox.create = AsyncMock(return_value="owned-test-box")
            server._sandbox.prepare = AsyncMock()
            server._sandbox.read_artifact = AsyncMock(return_value="<svg/>")
            server._sandbox.provider.serialize_handle = AsyncMock(
                return_value={"sandbox_id": "owned-test-box"}
            )
            server._sandbox.provider.close = AsyncMock()
            response.status = None  # The native harness adapters may omit status.
            for harness in ("opencode", "codex", "hermes"):
                row = gym_row(Task("mmsvg_text2svg", "case", "Draw a circle"), harness)
                seed = await client.post("/seed_session", json=row)
                assert seed.status_code == 200, seed.text
                assert seed.json()["sandbox_descriptor"]["workdir"] == "/workspace"
                verified = await client.post(
                    "/verify", json=row | {"response": response.model_dump(mode="json")}
                )
                assert verified.status_code == 200, verified.text
                assert verified.json()["reward"] == 0.76
                assert not server._sessions
            assert server._sandbox.provider.close.await_count == 3

    with tempfile.TemporaryDirectory() as directory:
        server.config.verification_log_dir = directory
        asyncio.run(roundtrip())
        records = [json.loads(path.read_text()) for path in Path(directory).glob("*/record.json")]
        assert records and all(record["complete"] for record in records)
        assert any(record["result"].get("mask_sample") for record in records)
        assert any(record["result"].get("reward") == 0.76 for record in records)
        assert all(record["response_id"] == "resp_test" for record in records)
        assert all("failure_reason" not in record["result"] for record in records)
        assert any(record.get("harness_failure_code") == "codex_nonzero_exit" for record in records)
        assert any(record.get("harness_failure_code") == "unclassified" for record in records)
        assert "provider unavailable" not in json.dumps(records)
    print(
        "Native Gym configs, discovery, capture settings and HTTP lifecycle: PASS",
        flush=True,
    )
    if base_path := os.environ.get("MMRL_BASE_CONFIG"):
        from nemo_rl.algorithms.single_controller_utils import MasterConfig
        from nemo_rl.algorithms.single_controller_utils.config import (
            validate_single_controller_config,
        )

        cfg = training_config(
            json.loads(Path(base_path).read_text()),
            "/tmp/mm-gym.json",
            "/tmp/mm-data.jsonl",
            "/tmp/mm-config",
        )
        validate_single_controller_config(MasterConfig(**cfg))
        print("Native V2 composed configuration: PASS (no GPU update)", flush=True)


if __name__ == "__main__":
    main()
