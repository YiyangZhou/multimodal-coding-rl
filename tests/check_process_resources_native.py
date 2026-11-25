"""Real CodingResources lifecycle and native MCP; synthetic renderer/judge."""
import argparse
from contextlib import asynccontextmanager
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def check(gym, patched):
    load("nemo_gym.mcp_auto_exposure", patched / "nemo_gym/mcp_auto_exposure.py")
    native = load("native_mcp_fixture", gym / "tests/unit_tests/test_mcp_auto_exposure.py")
    capture = load("native_capture_fixture", gym / "tests/unit_tests/test_token_capture_staging_rebuild.py")
    from fastapi.testclient import TestClient
    from nemo_gym.openai_utils import NeMoGymResponse
    from nemo_gym.server_utils import ServerClient
    from nemo_gym.token_id_capture.staging.records import RolloutManifest
    from mmrl.process_gym_app import ProcessCodingResourcesServer, ProcessConfig
    from mmrl.process_rewards import ProcessRewardConfig
    from mmrl.reward import InfrastructureError, Quality
    from mmrl.tasks import Task, gym_row

    receipt, _ = capture._branched()
    root = next(r for r in receipt.manifest if r.model_call_id == "root")
    async def manifest(self, rollout):
        return RolloutManifest(rollout_id=rollout, records=receipt.manifest)
    @asynccontextmanager
    async def no_gateway(self):
        yield
    final = NeMoGymResponse(
        id=root.response_id, created_at=1.0, model="policy", object="response", status="completed",
        parallel_tool_calls=True, tool_choice="auto", tools=[], output=[{
            "type": "message", "id": "final", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": "<svg/>", "annotations": []}]}])
    observed_tool = json.dumps({"id": root.response_id, "status": "completed", "output": [{
        "type": "function_call", "call_id": "preview-1", "namespace": "mcp__mmrl",
        "name": "preview", "arguments": "{}"}]}).encode()
    with tempfile.TemporaryDirectory() as directory, patch.dict("os.environ", {
            "MMRL_CAPTURE_GATEWAY_TOKEN": "synthetic-process-session-gateway-token"}):
        with patch("mmrl.gym_app.RemoteSandbox"), patch("mmrl.gym_app.QualityJudge"):
            server = ProcessCodingResourcesServer(
                config=ProcessConfig(host="127.0.0.1", port=18004, name="mm_coding", entrypoint="process_app.py",
                    process_reward_config={"progress_weight": 0.04},
                    sandbox_image="example.invalid/test@sha256:" + "a"*64, capture_gateway_port=18443,
                    verification_log_dir=directory),
                server_client=MagicMock(spec=ServerClient, global_config_dict={}))
        sandbox, judge = server._sandbox, server._service.judge
        sandbox.create = AsyncMock(return_value="owned-synthetic-box")
        sandbox.prepare = AsyncMock()
        sandbox.read_artifact = AsyncMock(return_value="<svg/>")
        sandbox.render = AsyncMock(return_value=b"synthetic-independent-preview")
        sandbox.provider.serialize_handle = AsyncMock(return_value={"sandbox_id": "owned-synthetic-box"})
        sandbox.provider.upload_file = AsyncMock()
        sandbox.provider.close = AsyncMock()
        sandbox.provider.aclose = AsyncMock()
        judge.grade = AsyncMock(return_value=Quality(0.8, 0.6))
        with patch.object(ProcessCodingResourcesServer, "_manifest", manifest), patch.object(
                ProcessCodingResourcesServer, "_gateway_context", no_gateway):
            app = server.setup_webserver()
            native.maybe_auto_expose(server, app)
            with TestClient(app) as client:
                native._handshake(client)
                for index, failed, route in ((0, False, "codex"), (1, True, "hermes"), (2, False, "single")):
                    rollout = f"{receipt.rollout_id}-{index}"
                    body = gym_row(Task("mmsvg_text2svg", "case", "Draw a circle"), route) | {"_ng_rollout_id": rollout}
                    seed = client.post("/seed_session", json=body)
                    assert seed.status_code == 200, seed.text[:300]
                    metadata = seed.json()
                    token = metadata["mcp"]["headers"][native.TOKEN_HEADER]
                    if route != "single":
                        assert metadata["mmrl_preview_url"] == "https://127.0.0.1:18443/mcp"
                        assert metadata["mmrl_preview_headers"]["Authorization"].startswith("Bearer ")
                        async def observe():
                            await server._observe_response(
                                f"/ng-rollout/{rollout}/training-token-capture/v1/responses",
                                "application/json", observed_tool)
                        client.portal.call(observe)
                        judge.grade.side_effect = InfrastructureError("synthetic API failure") if failed else None
                        preview = native._call(client, "preview", {}, token)
                        if failed:
                            assert preview.get("isError") is True
                        else:
                            assert native._payload(preview)["executed"]
                    else:
                        assert native._list(client, token) == []
                        judge.grade.side_effect = None
                    before = judge.grade.await_count
                    reply = client.post("/verify", json=body | {"response": final.model_dump(mode="json")})
                    assert reply.status_code == 200, reply.text[:300]
                    result = reply.json()
                    assert result["mask_sample"] is failed
                    assert result["instance_config"]["mask_sample"] is failed
                    if failed:
                        assert judge.grade.await_count == before
                        assert result["failure_kind"] == "mm_coding:infrastructure"
                        assert "process_reward_payload" not in result
                    elif route != "single":
                        payload = result["process_reward_payload"]
                        assert payload["rollout_id"] == rollout and len(payload["stages"]) == 1
                        assert payload["stages"][0]["model_call_id"] == "root"
                        assert payload["final_reward"] == result["reward"]
                        assert set(payload["config"]) == set(ProcessRewardConfig.__dataclass_fields__)
                        assert payload["config"]["progress_weight"] == .04
                    else:
                        payload = result["process_reward_payload"]
                        assert payload["rollout_id"] == rollout and not payload["stages"]
                        assert payload["final_reward"] == result["reward"]
                        for key in ("progress_weight", "token_cost_per_1000", "render_cost", "duplicate_render_cost"):
                            assert payload["config"][key] == 0
                    assert not server._sessions and not server._capture_sessions
                assert sandbox.provider.close.await_count == 2
                # Lifespan shutdown must reclaim a seeded but unverified session.
                body = gym_row(Task("mmsvg_text2svg", "case", "Circle"), "opencode") | {"_ng_rollout_id": "shutdown-case"}
                assert client.post("/seed_session", json=body).status_code == 200
                assert len(server._sessions) == 1
            assert not server._sessions and not server._capture_sessions
            assert sandbox.provider.close.await_count == 3
            sandbox.provider.aclose.assert_awaited_once()
    return {"passed": True, "synthetic": True, "model_inference": False,
            "real_resource_seed_mcp_verify_close": True, "trusted_stage_payload_emitted": True,
            "unresolved_process_infrastructure_masks_episode_without_final_api": True,
            "single_turn_has_final_only_payload_and_no_tools": True,
            "unverified_session_cleaned_at_shutdown": True,
            "tls_gateway_and_real_harnesses": False, "learner_consumption": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gym-root", type=Path, required=True)
    parser.add_argument("--patched-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(args.gym_root)]
    report = check(args.gym_root, args.patched_root)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
