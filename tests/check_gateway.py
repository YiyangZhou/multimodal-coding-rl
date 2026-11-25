"""Real HTTP/SSE contract test; run with the pinned Gym aiohttp environment."""

import asyncio
import json
import socket
from unittest.mock import patch
from aiohttp import ClientSession, web
from mmrl.gateway import create_app, _keep_connection_alive


async def main():
    calls = []
    observed_socket_options = []
    def observe_keepalive(transport):
        _keep_connection_alive(transport)
        sock = transport.get_extra_info("socket")
        values = [sock.getsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE)]
        for name in ("TCP_KEEPIDLE", "TCP_KEEPINTVL", "TCP_KEEPCNT"):
            values.append(sock.getsockopt(socket.IPPROTO_TCP, getattr(socket, name)))
        observed_socket_options.append(values)
    first_frame = asyncio.Event()
    release_tail = asyncio.Event()
    frames = [
        b'data: {"token_ids":[3,4],"logprobs":[-0.1,-0.2]}\n\n',
        b"data: [DONE]\n\n",
    ]

    async def source(request):
        calls.append((request.path, await request.read()))
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await response.write(frames[0])
        first_frame.set()
        await release_tail.wait()
        await response.write(frames[1])
        await response.write_eof()
        return response

    async def start(app):
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        return runner, f"http://127.0.0.1:{runner.addresses[0][1]}"

    backend = web.Application()
    backend.router.add_route("*", "/{path:.*}", source)
    back_runner, origin = await start(backend)
    token = "test-only-gateway-token-with-32-characters"
    proxy_runner, proxy = await start(create_app(origin, token))
    path = "/ng-rollout/case-1/training-token-capture/v1/chat/completions"
    payload = b'{"model":"policy","messages":[],"stream":true}'
    try:
        with patch("mmrl.gateway._keep_connection_alive", side_effect=observe_keepalive):
            async with ClientSession() as client:
                async with client.post(proxy + path, data=payload) as response:
                    assert response.status == 401
                headers = {
                    "Authorization": "Bearer " + token,
                    "Content-Type": "application/json",
                }
                for forbidden in (
                    "/v1/chat/completions",
                    "/training-token-capture/freeze",
                    path + "?upstream=other",
                ):
                    async with client.post(
                        proxy + forbidden, headers=headers, data=payload
                    ) as response:
                        assert response.status == 404
                assert not calls
                async with client.post(
                    proxy + path, headers=headers, data=payload
                ) as response:
                    assert response.status == 200
                    await asyncio.wait_for(first_frame.wait(), 5)
                    # The first SSE frame must arrive before the backend is allowed to finish.
                    first = await asyncio.wait_for(
                        response.content.readexactly(len(frames[0])), 5
                    )
                    assert first == frames[0]
                    release_tail.set()
                    assert await response.read() == frames[1]
                assert calls == [(path, payload)]
        assert observed_socket_options == [[1, 60, 20, 3]]
    finally:
        release_tail.set()
        await proxy_runner.cleanup()
        await back_runner.cleanup()
    print(
        json.dumps(
            {
                "gateway_http_sse": "passed",
                "accepted_socket_keepalive": observed_socket_options,
                "auth_required": True,
                "control_routes_blocked": True,
                "capture_path_preserved": True,
                "body_preserved": True,
                "sse_not_buffered": True,
                "tls_remote_test": "pending",
            }
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
