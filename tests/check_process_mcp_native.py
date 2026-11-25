"""Exercise the preview chain through signed native Gym MCP sessions.

Uses actual MCP HTTP dispatch and a synthetic renderer/judge; no model inference.
Loads the optional MCP patch from a separate source tree, never the active Gym.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4


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
    from fastapi import Request
    from fastapi.testclient import TestClient
    from nemo_gym.base_resources_server import BaseSeedSessionRequest, BaseSeedSessionResponse, SimpleResourcesServer
    from nemo_gym.server_utils import SESSION_ID_KEY
    from nemo_gym.token_id_capture.staging.records import RolloutManifest
    from mmrl.process_observer import PreviewCallObserver
    from mmrl.process_stages import StageCollector
    from mmrl.process_tool import PreviewSession, install_preview_route
    from mmrl.reward import InfrastructureError, Quality
    from mmrl.service import CodingService
    from mmrl.tasks import Task

    receipt, _ = capture._branched()
    manifest = RolloutManifest(rollout_id=receipt.rollout_id, records=receipt.manifest)
    records = {r.model_call_id: r for r in manifest.records}
    async def get_manifest(rollout):
        assert rollout == manifest.rollout_id
        return manifest

    def response(call):
        return json.dumps({"id": records[call].response_id, "status": "completed", "output": [
            {"type": "function_call", "call_id": "tool-" + call,
             "namespace": "mmrl", "name": "preview", "arguments": "{}"}]}).encode()

    sessions, seeded = {}, []
    with tempfile.TemporaryDirectory() as directory:
        class Resources(SimpleResourcesServer):
            async def seed_session(self, request: Request, body: BaseSeedSessionRequest):
                sid = uuid4().hex
                request.session[SESSION_ID_KEY] = sid
                sandbox, judge = AsyncMock(), AsyncMock()
                sandbox.read_artifact.return_value = "<svg/>"
                sandbox.render.return_value = b"synthetic-independent-image"
                judge.grade.return_value = Quality(0.8, 0.6)
                collector = StageCollector(
                    rollout_id=receipt.rollout_id, task=Task("mmsvg_text2svg", "case", "Circle"),
                    route="codex", sandbox=sandbox, service=CodingService(sandbox, judge),
                    render_conditions={"image": "synthetic-pinned-image", "judge": "synthetic"},
                    audit_directory=Path(directory) / sid)
                observer = PreviewCallObserver(receipt.rollout_id, get_manifest, tool_names={"mmrl__preview"})
                session = PreviewSession(observer, collector, "synthetic-handle", "mmrl__preview")
                sessions[sid] = session
                if not seeded:
                    await observer(f"/ng-rollout/{receipt.rollout_id}/training-token-capture/v1/responses",
                                   "application/json", response("root"))
                seeded.append(SimpleNamespace(session=session, sandbox=sandbox, judge=judge))
                return BaseSeedSessionResponse()

            async def verify(self, body):
                pass

            def setup_webserver(self):
                app = super().setup_webserver()
                install_preview_route(app, sessions.get)
                return app

        server = native._server(Resources, name="mmrl")
        app = server.setup_webserver()
        native.maybe_auto_expose(server, app)
        with TestClient(app) as client:
            token = native._seed(client)
            native._handshake(client)
            assert [t["name"] for t in native._list(client, token)] == ["preview"]
            # Even a valid session cookie cannot forge the SDK transport state.
            assert client.post("/preview", json={}).status_code == 403
            assert native._call(client, "preview", {}, token="forged").get("isError") is True
            result = native._payload(native._call(client, "preview", {}, token))
            assert set(result) == {"executed", "preview_path", "message"}
            assert result["executed"] and result["preview_path"].startswith("/workspace/.mmrl-preview-")
            first = seeded[0]
            assert first.judge.grade.await_count == 1 and first.sandbox.render.await_count == 1
            upload = first.sandbox.provider.upload_file.call_args.args
            assert upload[0] == "synthetic-handle" and upload[2] == result["preview_path"]
            assert upload[1].read_bytes() == b"synthetic-independent-image"
            assert native._payload(native._call(client, "preview", {}, token)) == result
            assert first.judge.grade.await_count == 1 and len(first.session.collector.stages) == 1
            # A different signed session cannot consume the first session's call.
            other_token = native._seed(client)
            assert native._call(client, "preview", {}, other_token).get("isError") is True
            assert seeded[1].judge.grade.await_count == 0
            assert seeded[1].session.unresolved_failures
            # Drive a second committed call on the same portal/loop as MCP.
            async def register_second():
                await first.session.observer(
                    f"/ng-rollout/{receipt.rollout_id}/training-token-capture/v1/responses",
                    "application/json", response("main"))
            client.portal.call(register_second)
            first.sandbox.read_artifact.return_value = "<svg>changed</svg>"
            first.judge.grade.side_effect = InfrastructureError("private upstream detail")
            args = {"name": "preview", "arguments": {}}
            failed = native._rpc(client, "tools/call", args, token=token, rid=4)["result"]
            assert failed.get("isError") is True and "private upstream detail" not in json.dumps(failed)
            assert first.session.unresolved_failures
            assert len(first.session.collector.stages) == 1
            first.judge.grade.side_effect = None
            retried = native._payload(native._rpc(client, "tools/call", args, token=token, rid=4)["result"])
            assert retried["executed"] and not first.session.unresolved_failures
            assert len(first.session.collector.stages) == 2
            assert first.judge.grade.await_count == 3  # one original, one failure, one successful retry
    return {"passed": True, "synthetic": True, "model_inference": False,
            "native_signed_mcp_to_observer_to_collector_to_image_upload": True,
            "sdk_request_id_idempotency": True, "cross_session_call_rejected": True,
            "private_scores_absent_from_tool_feedback": True,
            "api_failure_unsealed_until_same_request_recovers": True,
            "active_runtime_changed": False, "learner_consumption": False}


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
