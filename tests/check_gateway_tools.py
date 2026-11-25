"""Actual HTTP test of the optional, fixed-origin MCP gateway route."""
import asyncio
import json
from aiohttp import ClientSession, web
from mmrl.gateway import create_app


async def main():
    model_calls, tool_calls, observations = [], [], []
    async def model(request):
        model_calls.append(request.path)
        return web.Response(body=b'{"id":"served"}', content_type="application/json")
    async def tool(request):
        tool_calls.append((request.method, request.path, await request.read(),
                           request.headers.get("X-NeMo-Gym-Session-Token"),
                           request.headers.get("Authorization")))
        return web.Response(body=b'{"result":{"ok":true}}', content_type="application/json",
                            headers={"Mcp-Session-Id": "synthetic-session"})
    async def observe(*args):
        observations.append(args)
    async def start(handler):
        app = web.Application()
        app.router.add_route("*", "/{path:.*}", handler)
        return await start_app(app)
    async def start_app(app):
        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", 0).start()
        return runner, f"http://127.0.0.1:{runner.addresses[0][1]}"
    mr, mo = await start(model)
    tr, to = await start(tool)
    token = "test-only-mcp-routing-gateway-token"
    gr, go = await start_app(create_app(mo, token, response_observer=observe, tool_upstream=to))
    body = b'{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{}}'
    headers = {"Authorization": "Bearer " + token, "X-NeMo-Gym-Session-Token": "synthetic-signed-token"}
    try:
        async with ClientSession() as client:
            async with client.post(go + "/mcp", data=body) as response:
                assert response.status == 401
            for path in ("/verify", "/seed_session", "/mcp/other", "/mcp?upstream=other",
                         "/training-token-capture/control/rollouts/x/manifest"):
                async with client.post(go + path, data=body, headers=headers) as response:
                    assert response.status == 404
            async with client.post(go + "/mcp", data=body, headers=headers) as response:
                assert response.status == 200 and response.headers["Mcp-Session-Id"] == "synthetic-session"
                assert await response.read() == b'{"result":{"ok":true}}'
            assert tool_calls == [("POST", "/mcp", body, "synthetic-signed-token", None)]
            assert model_calls == [] and observations == []
            path = "/ng-rollout/test/training-token-capture/v1/responses"
            async with client.post(go + path, data=b'{}', headers=headers) as response:
                assert response.status == 200
                assert await response.read() == b'{"id":"served"}'
            assert model_calls == [path] and len(observations) == 1 and len(tool_calls) == 1
    finally:
        await gr.cleanup()
        await tr.cleanup()
        await mr.cleanup()
    print(json.dumps({"passed": True, "fixed_mcp_origin_only": True,
                      "signed_session_header_preserved": True, "gateway_bearer_not_forwarded": True,
                      "model_observer_not_applied_to_mcp": True, "resource_control_routes_blocked": True}))


if __name__ == "__main__":
    asyncio.run(main())
