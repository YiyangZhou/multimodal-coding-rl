"""Native Gym final answers must pass the process gateway without tool calls."""
import asyncio
import json

from aiohttp import ClientSession, web
from nemo_gym.openai_utils import NeMoGymChatCompletion

from mmrl.gateway import create_app
from mmrl.process_observer import PreviewCallObserver


async def main():
    async def no_manifest(rollout_id):
        raise AssertionError("A final text answer must not claim preview evidence")

    observer = PreviewCallObserver("final-answer", no_manifest, tool_names={"mmrl__preview"})
    path = "/ng-rollout/final-answer/training-token-capture/v1/chat/completions"
    token = "synthetic-final-answer-gateway-token"
    base = {"id": "chatcmpl-final", "object": "chat.completion", "created": 1,
            "model": "policy", "choices": [{"index": 0, "finish_reason": "stop",
            "message": {"role": "assistant", "content": "The answer is saved."}}]}
    payloads = []
    for label, fields in (("omitted", {}), ("null", {"tool_calls": None}),
                          ("empty", {"tool_calls": []})):
        raw = json.loads(json.dumps(base))
        raw["choices"][0]["message"].update(fields)
        # The same typed response and JSON serialization used by Gym's route.
        typed = NeMoGymChatCompletion.model_validate(raw).model_dump(mode="json")
        payloads.append((label, json.dumps(typed).encode()))

    current = payloads[0][1]

    async def source(request):
        return web.Response(body=current, content_type="application/json")

    async def start(app):
        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", 0).start()
        return runner, f"http://127.0.0.1:{runner.addresses[0][1]}"

    backend = web.Application()
    backend.router.add_post(path, source)
    upstream_runner, upstream = await start(backend)
    runner, origin = await start(create_app(upstream, token, response_observer=observer))
    try:
        async with ClientSession() as client:
            for label, current in payloads:
                async with client.post(origin + path, json={},
                                       headers={"Authorization": "Bearer " + token}) as response:
                    body = await response.read()
                    assert response.status == 200, (label, response.status, body)
                    assert body == current, "Gateway changed the model answer"
            for invalid in (False, "", {}, 1):
                raw = json.loads(json.dumps(base))
                raw["choices"][0]["message"]["tool_calls"] = invalid
                current = json.dumps(raw).encode()
                async with client.post(origin + path, json={},
                                       headers={"Authorization": "Bearer " + token}) as response:
                    assert response.status == 502, "Malformed tool collection was accepted"
    finally:
        await runner.cleanup()
        await upstream_runner.cleanup()
    print(json.dumps({"passed": True, "valid_final_answer_cases": len(payloads),
                      "malformed_cases_rejected": 4, "real_http": True,
                      "native_gym_serialization": True, "model_inference": False}))


if __name__ == "__main__":
    asyncio.run(main())
