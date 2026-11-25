"""Authenticated TLS forwarding to one Gym model server, preserving capture paths."""

import argparse
import hmac
import ipaddress
import os
import re
import socket
import ssl
from urllib.parse import urlsplit

# Never forward Gym's capture-control, file, lifecycle or arbitrary proxy routes.
CAPTURE_PATH = re.compile(
    r"/ng-rollout/[A-Za-z0-9_.-]{1,160}/training-token-capture/v1/"
    r"(?:responses|chat/completions|completions|models)"
)
HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "host",
    "authorization",
}


def _keep_connection_alive(transport):
    """Keep silent generation requests alive across idle network devices."""
    sock = transport.get_extra_info("socket") if transport is not None else None
    if sock is None:
        return
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    for name, value in (("TCP_KEEPIDLE", 60), ("TCP_KEEPINTVL", 20), ("TCP_KEEPCNT", 3)):
        option = getattr(socket, name, None)
        if option is not None:
            sock.setsockopt(socket.IPPROTO_TCP, option, value)


def create_app(upstream, token, *, response_observer=None, observation_limit=64 * 1024 * 1024,
               tool_upstream=None):
    """Optionally register trusted response evidence before releasing it to a CLI.

    The observer receives the rollout-scoped path, content type and exact bytes
    served by Gym, never an agent-submitted call identity. It must finish before
    the harness can execute any returned tool calls. Observation is opt-in: the
    normal path continues streaming without buffering. The callback must reject
    incomplete/ambiguous evidence rather than assign it to the latest call.
    """
    import aiohttp
    from aiohttp import web

    parsed = urlsplit(upstream)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.path not in {"", "/"}
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Upstream must be one explicit Gym model-server origin")
    if not token or len(token) < 32:
        raise ValueError(
            "Use a dedicated random gateway token of at least 32 characters"
        )
    if type(observation_limit) is not int or observation_limit <= 0:
        raise ValueError("Response observation needs a positive byte limit")
    upstream = upstream.rstrip("/")
    if tool_upstream is not None:
        target = urlsplit(tool_upstream)
        if (target.scheme not in {"http", "https"} or not target.hostname
                or target.path not in {"", "/"} or target.username or target.password
                or target.query or target.fragment):
            raise ValueError("Tool upstream must be one explicit resources-server origin")
        tool_upstream = tool_upstream.rstrip("/")
    client_key = web.AppKey("client", aiohttp.ClientSession)

    async def lifecycle(app):
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(
                total=12000, sock_connect=30, sock_read=11400
            ),
            auto_decompress=False,
            connector=aiohttp.TCPConnector(limit=128),
            trust_env=False,
        ) as client:
            app[client_key] = client
            yield

    async def forward(request):
        supplied = request.headers.get("Authorization", "")
        if not hmac.compare_digest(supplied.encode(), ("Bearer " + token).encode()):
            raise web.HTTPUnauthorized()
        path = request.path
        is_tool = tool_upstream is not None and path == "/mcp"
        if request.query_string or not (is_tool or CAPTURE_PATH.fullmatch(path)):
            raise web.HTTPNotFound()
        methods = ["GET", "POST", "DELETE"] if is_tool else ["GET"] if path.endswith("/models") else ["POST"]
        if request.method not in methods:
            raise web.HTTPMethodNotAllowed(request.method, methods)
        # Bound the request before opening upstream; do not consume unbounded
        # uploads or rewrite messages, image bytes, token IDs, or SSE frames.
        _keep_connection_alive(request.transport)
        body = await request.read()
        blocked = HOP_HEADERS | {
            x.strip().lower() for x in request.headers.get("Connection", "").split(",")
        }
        headers = {k: v for k, v in request.headers.items() if k.lower() not in blocked}
        observing = response_observer is not None and request.method == "POST" and not is_tool
        if observing:
            # Inspect the actual wire payload without a decompression bomb or
            # dependence on the CLI's advertised encodings.
            headers = {k: v for k, v in headers.items() if k.lower() != "accept-encoding"}
            headers["Accept-Encoding"] = "identity"
        response = None
        try:
            async with request.app[client_key].request(
                request.method,
                (tool_upstream if is_tool else upstream) + path,
                data=body,
                headers=headers,
                allow_redirects=False,
            ) as remote:
                blocked_response = HOP_HEADERS | {
                    x.strip().lower()
                    for x in remote.headers.get("Connection", "").split(",")
                }
                buffered = None
                if observing and remote.status == 200:
                    if remote.headers.get("Content-Encoding", "identity").lower() != "identity":
                        raise web.HTTPBadGateway(text="Unsupported observed response encoding")
                    buffered = bytearray()
                    async for chunk in remote.content.iter_any():
                        if len(buffered) + len(chunk) > observation_limit:
                            raise web.HTTPBadGateway(text="Observed response exceeds byte limit")
                        buffered.extend(chunk)
                    buffered = bytes(buffered)
                    try:
                        await response_observer(path, remote.headers.get("Content-Type", ""), buffered)
                    except Exception:
                        # Do not expose partial model output or callback details.
                        # A failed evidence commit is infrastructure failure,
                        # never a successful tool call with an invented score.
                        raise web.HTTPBadGateway(text="Response evidence could not be registered") from None
                response = web.StreamResponse(
                    status=remote.status,
                    headers={
                        k: v
                        for k, v in remote.headers.items()
                        if k.lower() not in blocked_response
                    },
                )
                await response.prepare(request)
                if buffered is not None:
                    await response.write(buffered)
                else:
                    async for chunk in remote.content.iter_any():
                        await response.write(chunk)
                await response.write_eof()
                return response
        except (aiohttp.ClientError, TimeoutError):
            if response is not None and response.prepared:
                # A broken SSE response must remain incomplete, never be made
                # to look like a successful shortened model response.
                if request.transport:
                    request.transport.close()
                return response
            raise web.HTTPBadGateway(text="Gym model server unavailable") from None

    app = web.Application(client_max_size=64 * 1024 * 1024)
    app.cleanup_ctx.append(lifecycle)
    app.router.add_route("*", "/{path:.*}", forward)
    return app


def main():
    from aiohttp import web

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--upstream", required=True, help="Gym policy_model origin; never raw vLLM"
    )
    parser.add_argument("--listen-host", required=True, help="Private gateway bind address")
    parser.add_argument("--port", type=int, default=18443)
    parser.add_argument("--cert", required=True)
    parser.add_argument("--key", required=True)
    args = parser.parse_args()
    address = ipaddress.ip_address(args.listen_host)
    if not address.is_private or address.is_unspecified or address.is_multicast:
        raise SystemExit("Bind to a specific private interface")
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.load_cert_chain(args.cert, args.key)
    app = create_app(args.upstream, os.environ.get("MMRL_CAPTURE_GATEWAY_TOKEN", ""))
    # Access logging is disabled: rollout-scoped URLs should not enter shared logs.
    web.run_app(
        app, host=args.listen_host, port=args.port, ssl_context=context, access_log=None
    )


if __name__ == "__main__":
    main()
