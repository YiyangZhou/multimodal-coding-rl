"""Native Gym SSE/manifest plus actual HTTP; synthetic responses, no inference."""
import argparse
import asyncio
import importlib.util
import json
from pathlib import Path
import sys


async def check(gym_root):
    from aiohttp import ClientSession, web
    from nemo_gym.chat_streaming import synthesize_chat_completion_sse
    from nemo_gym.responses_streaming import synthesize_responses_sse
    from nemo_gym.token_id_capture.staging.records import RolloutManifest
    from mmrl.gateway import create_app
    from mmrl.process_observer import PreviewCallObserver
    from mmrl.reward import InfrastructureError

    spec = importlib.util.spec_from_file_location(
        "native_fixture", gym_root / "tests/unit_tests/test_token_capture_staging_rebuild.py")
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    receipt, _ = fixture._branched()
    manifest = RolloutManifest(rollout_id=receipt.rollout_id, records=receipt.manifest)
    record = next(r for r in manifest.records if r.model_call_id == "root")
    async def getter(rollout_id):
        assert rollout_id == receipt.rollout_id
        return manifest

    def observer():
        return PreviewCallObserver(receipt.rollout_id, getter, tool_names={"mmrl__preview"})

    item = {"type": "function_call", "call_id": "tool-1", "namespace": "mmrl",
            "name": "preview", "arguments": '{"view":"default"}'}
    response = {"id": record.response_id, "object": "response", "status": "completed", "output": [item]}
    raw = "".join(synthesize_responses_sse(response)).encode()
    path = f"/ng-rollout/{receipt.rollout_id}/training-token-capture/v1/responses"
    chat_path = path.removesuffix("responses") + "chat/completions"
    completed = {"id": record.response_id, "object": "chat.completion", "created": 1, "model": "policy",
                 "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                     "role": "assistant", "content": None, "tool_calls": [
                         {"type": "function", "id": "tool-1", "function": {
                             "name": "mmrl__preview", "arguments": item["arguments"]}}]}}]}
    chat_raw = "".join(synthesize_chat_completion_sse(completed)).encode()

    async def reject(awaitable, kinds=(ValueError, InfrastructureError)):
        try:
            await awaitable
        except kinds:
            return
        raise AssertionError("Invalid evidence was accepted")

    for route, mime, data in ((path, "text/event-stream", raw),
                              (path, "application/json", json.dumps(response).encode()),
                              (chat_path, "text/event-stream", chat_raw),
                              (chat_path, "application/json", json.dumps(completed).encode())):
        current = observer()
        await reject(current.claim("unobserved", "mmrl__preview", {"view": "default"}))
        await current(route, mime, data)
        call = await current.claim("request-1", "mmrl__preview", {"view": "default"})
        assert call.model_call_id == record.model_call_id and call.response_id == record.response_id
        assert call.rollout_id == receipt.rollout_id and call.tool_call_id == "tool-1"
        await current(route, mime, data)  # Response retransmission is not another render.
        assert call == await current.claim("request-1", "mmrl__preview", {"view": "default"})
        await reject(current.claim("request-2", "mmrl__preview", {"view": "default"}))
        await reject(current.claim("request-1", "mmrl__preview", {"view": "changed"}))
        next_record = next(r for r in manifest.records if r.model_call_id == "main")
        next_response = response | {"id": next_record.response_id,
                                    "output": [item | {"call_id": "tool-2"}]}
        await current(path, "application/json", json.dumps(next_response).encode())
        await reject(current.claim("request-1", "mmrl__preview", {"view": "default"}))
        assert (await current.claim("request-2", "mmrl__preview", {"view": "default"})).model_call_id == "main"

    current = observer()
    await reject(current(path.replace(receipt.rollout_id, "foreign"), "text/event-stream", raw))
    await reject(current(path, "text/event-stream", raw.rsplit(b"event: response.completed", 1)[0]))
    await reject(current(chat_path, "text/event-stream", chat_raw.replace(b"data: [DONE]\n\n", b"")))
    await reject(current(chat_path, "text/event-stream", chat_raw.replace(b'"index": 0', b'"index": 1')))
    await reject(current(path, "application/json", json.dumps(response | {"id": "uncommitted"}).encode()))
    ambiguous = response | {"output": [item, item | {"call_id": "tool-2"}]}
    await current(path, "application/json", json.dumps(ambiguous).encode())
    await reject(current.claim("ambiguous", "mmrl__preview", {"view": "default"}))
    await reject(current(path, "application/json", json.dumps(
        response | {"output": [item | {"arguments": '{"view":"changed"}'}]}).encode()))
    malformed = observer()
    await malformed(path, "application/json", json.dumps(response | {
        "output": [item | {"arguments": "malformed model arguments"}]}).encode())
    await reject(malformed.claim("invalid-arguments", "mmrl__preview", {}))

    async def start(app):
        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", 0).start()
        return runner, f"http://127.0.0.1:{runner.addresses[0][1]}"

    backend = web.Application()
    async def source(request):
        return web.Response(body=raw, content_type="text/event-stream")
    backend.router.add_post(path, source)
    upstream_runner, upstream = await start(backend)
    token = "synthetic-only-process-observer-token"
    current = observer()
    runner, origin = await start(create_app(upstream, token, response_observer=current))
    try:
        async with ClientSession() as client:
            async with client.post(origin + path, json={}, headers={"Authorization": "Bearer " + token}) as result:
                assert result.status == 200
                assert await result.read() == raw
        call = await current.claim(1, "mmrl__preview", {"view": "default"})
        assert call.model_call_id == "root"
    finally:
        await runner.cleanup()
        await upstream_runner.cleanup()
    return {"passed": True, "synthetic": True, "model_inference": False,
            "native_dialects": ["responses JSON", "responses SSE", "chat JSON", "chat SSE"],
            "real_http_gateway_to_native_manifest_join": True,
            "selected_model_call": call.model_call_id,
            "uncommitted_foreign_incomplete_ambiguous_and_reused_calls_rejected": True,
            "runtime_enabled": False,
            "scope": "Gateway response-to-ledger association only; this check does not execute a harness, renderer, judge or learner."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gym-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(args.gym_root)]
    report = asyncio.run(check(args.gym_root))
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
