"""Native seed/run/config propagation; remote CLI execution is mocked."""
import argparse
import asyncio
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import AsyncMock, MagicMock, patch


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def configuration_only(cls, config):
    # Pydantic model_construct invokes model_post_init and would install CLIs.
    # Only exercise native config serialization here; actual CLI probes are separate.
    instance = object.__new__(cls)
    object.__setattr__(instance, "__dict__", {"config": config})
    return instance


async def check(gym, patched):
    module = "responses_api_agents.harness_agent.app"
    harness = load(module, patched / "responses_api_agents/harness_agent/app.py")
    native = load("native_harness_fixture", gym / "responses_api_agents/harness_agent/tests/test_app.py")
    hermes = load("responses_api_agents.hermes_agent.app", patched / "responses_api_agents/hermes_agent/app.py")
    from responses_api_agents.codex_agent.app import CodexAgent, CodexAgentConfig
    from responses_api_agents.opencode_agent.app import OpenCodeAgent, OpenCodeAgentConfig
    from nemo_gym.sandbox.providers.base import SandboxExecResult
    from mmrl.harness_tools import configure_preview_tools, SESSION_HEADER
    import yaml

    gateway = "https://127.0.0.1:18443"
    examples = {
        "codex": {"codex_version": "0.144.4", "model": "policy"},
        "opencode": {"model": "policy", "opencode_config": {"permission": {"bash": "allow"}}},
        "hermes": {"model": "policy", "enabled_toolsets": ["terminal"]},
    }
    for name, original in examples.items():
        baseline = deepcopy(original)
        agent = native._make_agent(agent=name, agent_kwargs=original, sandbox_model_base_url=gateway)
        for index in range(2):
            seed = {
                "sandbox_descriptor": {"sandbox_id": "test-box", "workdir": "/workspace"},
                "mmrl_preview_url": gateway + "/mcp",
                "mmrl_preview_headers": {"Authorization": f"Bearer synthetic-{index}"},
                "mcp": {"transport": "http", "headers": {SESSION_HEADER: f"synthetic-session-{index}"}},
            }
            body = harness.HarnessAgentRunRequest(
                responses_create_params=native.NeMoGymResponseCreateParamsNonStreaming(input="Draw a circle"),
                _ng_rollout_id=f"synthetic-{name}-{index}",
            )
            response = native._response()
            verify = body.model_dump() | {"response": response, "reward": 0.7}
            agent.server_client.post = AsyncMock(side_effect=[MagicMock(cookies={}), MagicMock(cookies={})])
            agent._provider.exec = AsyncMock(side_effect=[
                SandboxExecResult(return_code=0, stdout="", stderr=""),
                SandboxExecResult(return_code=0, stdout="RUNNER_DONE", stderr=""),
            ])
            staged = {}
            async def provision(self, image, files, model_url):
                staged.update(files)
                return MagicMock(provider_name="opensandbox")
            with (
                patch.object(harness.HarnessAgent, "_runner", return_value=("# synthetic runner", {}, "python runner.py")),
                patch.object(harness.HarnessAgent, "_provision_box", provision),
                patch.object(harness.HarnessAgent, "_download_json", new=AsyncMock(return_value=response)),
                patch.object(harness.HarnessAgent, "_sandbox_model_url", return_value=gateway),
                patch(module + ".raise_for_status", new=AsyncMock()),
                patch(module + ".get_response_json", new=AsyncMock(side_effect=[seed, verify])),
            ):
                result = await agent.run(MagicMock(cookies={}), body)
            assert result.reward == 0.7
            captured = json.loads(staged["/work/agent_config.json"])
            assert "synthetic-session" not in staged["/work/request.json"]
            assert "mmrl_preview" not in staged["/work/request.json"]
            assert original == baseline and agent.config.agent_kwargs == baseline
            common = dict(host="127.0.0.1", port=18000, name=name, entrypoint="app.py")
            if name != "hermes":
                captured["model_server"] = None
            if name == "codex":
                cfg = CodexAgentConfig(**common, **captured)
                inner = configuration_only(CodexAgent, cfg)
                generated = inner._build_config(gateway)
                entry = generated["mcp_servers"]["mmrl"]
                headers = entry["http_headers"]
                assert entry["required"] and entry["enabled_tools"] == ["preview"]
                assert entry["tool_timeout_sec"] == 600
            elif name == "opencode":
                cfg = OpenCodeAgentConfig(**common, **captured)
                inner = configuration_only(OpenCodeAgent, cfg)
                generated = inner._build_opencode_config()
                entry = generated["mcp"]["mmrl"]
                headers = entry["headers"]
                assert entry["timeout"] == 600000 and entry["oauth"] is False
                assert generated["permission"] == {"bash": "allow", "mmrl_*": "allow"}
            else:
                cfg = hermes.HermesAgentConfig(**common, **captured)
                inner = configuration_only(hermes.HermesAgent, cfg)
                generated = yaml.safe_load(inner._build_config())
                entry = generated["mcp_servers"]["mmrl"]
                headers = entry["headers"]
                assert cfg.required_mcp_tools == ["mcp_mmrl_preview"]
                assert cfg.enabled_toolsets == ["terminal", "mcp-mmrl"]
                assert entry["timeout"] == 600 and entry["tools"]["include"] == ["preview"]
            assert headers == {"Authorization": f"Bearer synthetic-{index}", SESSION_HEADER: f"synthetic-session-{index}"}
            assert not harness._RUN_CONTEXT.get()
        assert configure_preview_tools(name, original, {}, gateway=gateway) == original
        for bad_url in ("https://elsewhere.invalid/mcp", gateway + "/verify", gateway + "/mcp?token=x"):
            invalid = seed | {"mmrl_preview_url": bad_url}
            try:
                configure_preview_tools(name, original, invalid, gateway=gateway)
            except ValueError:
                pass
            else:
                raise AssertionError("Invalid preview destination was accepted")
    # Existing non-preview path retains its exact native context and verification contract.
    await native.test_run_preserves_rollout_path_and_verifier_fields()
    return {"passed": True, "native_harness_seed_run_verify": True,
            "pinned_agent_config_serialization": ["codex", "opencode", "hermes"],
            "two_sessions_each_no_credential_leak_to_policy_or_next_session": True,
            "foreign_origin_and_non_mcp_path_rejected": True,
            "non_preview_native_regression": True,
            "remote_cli_execution": False, "model_inference": False, "learner_update": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gym-root", type=Path, required=True)
    parser.add_argument("--patched-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(args.gym_root)]
    result = asyncio.run(check(args.gym_root, args.patched_root))
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
