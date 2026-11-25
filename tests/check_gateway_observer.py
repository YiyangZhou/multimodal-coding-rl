"""Real HTTP ordering/failure tests for the opt-in trusted response observer."""
import asyncio
import json

from aiohttp import ClientSession, web

from mmrl.gateway import create_app


async def start(app):
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 0).start()
    return runner, f"http://127.0.0.1:{runner.addresses[0][1]}"


async def main():
    observations = []
    arrived, release = asyncio.Event(), asyncio.Event()
    payload = b'data: {"id":"served-1","choices":[]}\n\ndata: [DONE]\n\n'
    request_body = b'{"stream":true,"messages":[]}'
    paths = []

    async def source(request):
        assert request.headers["Accept-Encoding"] == "identity"
        assert await request.read() == request_body
        paths.append(request.path)
        if "/error/" in request.path:
            return web.Response(status=503, text="unavailable")
        if "/compressed/" in request.path:
            return web.Response(body=b"unused", headers={"Content-Encoding": "gzip"})
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await response.write(payload[:23])
        await asyncio.sleep(0)
        await response.write(payload[23:])
        await response.write_eof()
        return response

    async def observe(path, content_type, body):
        observations.append((path, content_type, body))
        if "/reject/" in path:
            raise ValueError("private diagnosis must not reach the agent")
        if "/ordered/" in path:
            arrived.set()
            await release.wait()

    backend = web.Application()
    backend.router.add_post("/{path:.*}", source)
    upstream_runner, upstream = await start(backend)
    token = "test-observer-only-token-32-characters"
    runner, origin = await start(create_app(upstream, token, response_observer=observe))
    small_runner, small_origin = await start(create_app(
        upstream, token, response_observer=observe, observation_limit=16))
    path = lambda name: f"/ng-rollout/{name}/training-token-capture/v1/chat/completions"
    headers = {"Authorization": "Bearer " + token, "Accept-Encoding": "gzip"}
    try:
        async with ClientSession() as client:
            pending = asyncio.create_task(client.post(origin + path("ordered"), data=request_body, headers=headers))
            await asyncio.wait_for(arrived.wait(), 5)
            await asyncio.sleep(0.03)
            assert not pending.done(), "CLI saw response before evidence registration completed"
            release.set()
            async with await asyncio.wait_for(pending, 5) as response:
                assert response.status == 200
                assert await response.read() == payload
            assert observations == [(path("ordered"), "text/event-stream", payload)]
            async with client.post(origin + path("reject"), data=request_body, headers=headers) as response:
                assert response.status == 502
                assert await response.text() == "Response evidence could not be registered"
            count = len(observations)
            for base, name, status in ((small_origin, "oversize", 502),
                                       (origin, "compressed", 502),
                                       (origin, "error", 503)):
                async with client.post(base + path(name), data=request_body, headers=headers) as response:
                    assert response.status == status
                    assert payload not in await response.read()
            assert len(observations) == count
            async with client.post(origin + path("unauthorized"), data=request_body) as response:
                assert response.status == 401
            assert not any("unauthorized" in p for p in paths)
    finally:
        release.set()
        await small_runner.cleanup()
        await runner.cleanup()
        await upstream_runner.cleanup()
    print(json.dumps({"gateway_observer_http": "passed", "exact_wire_bytes": True,
                      "registration_before_cli_execution": True, "failure_returns_502": True,
                      "bounded_response": True, "compressed_response_rejected": True,
                      "live_harness_hook_enabled": False}))


if __name__ == "__main__":
    asyncio.run(main())
