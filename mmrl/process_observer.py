"""Join served preview calls to Gym's committed ledger, without agent call IDs.

This is a gateway callback and a tool-admission registry. The runtime does not
install it yet. A session-owned tool endpoint must call claim before invoking
StageCollector, and finalization must still verify the selected terminal chain.
"""
import asyncio
from dataclasses import dataclass
import hashlib
import json
import re

from .reward import InfrastructureError


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _served_response(content_type, raw, dialect):
    if content_type.split(";", 1)[0].strip().lower() == "application/json":
        response = json.loads(raw)
        if not isinstance(response, dict):
            raise ValueError("Expected a model response object")
        return response
    if content_type.split(";", 1)[0].strip().lower() != "text/event-stream":
        raise ValueError("Unsupported response evidence format")
    # Gym's reconstruction helpers are best-effort logging utilities. Validate
    # complete frames and a terminal event before using their reconstruction.
    text = raw.decode("utf-8", errors="strict").replace("\r\n", "\n")
    if not text.endswith("\n\n"):
        raise ValueError("Incomplete SSE frame")
    events, done = [], False
    for block in text.split("\n\n"):
        fields = [line[5:].lstrip(" ") for line in block.splitlines() if line.startswith("data:")]
        if not fields:
            continue
        if done:
            raise ValueError("Data after SSE terminal marker")
        payload = "\n".join(fields)
        if payload == "[DONE]":
            done = True
            continue
        event = json.loads(payload)
        if not isinstance(event, dict):
            raise ValueError("Invalid SSE event")
        events.append(event)
    if dialect == "responses":
        terminals = [e for e in events if e.get("type") in {
            "response.completed", "response.incomplete", "response.failed"}]
        if len(terminals) != 1 or events[-1] is not terminals[0]:
            raise ValueError("Missing or ambiguous terminal response")
        response = terminals[0].get("response")
        if not isinstance(response, dict):
            raise ValueError("Missing terminal response envelope")
        identities = {e["response"]["id"] for e in events
                      if isinstance(e.get("response"), dict) and e["response"].get("id")}
        if len(identities) > 1:
            raise ValueError("Response identity changed within stream")
        return response
    if not done:
        raise ValueError("Missing chat stream terminal marker")
    ids = {e.get("id") for e in events if e.get("id")}
    choices = [c for e in events for c in e.get("choices", [])]
    if len(ids) != 1 or not choices or any(c.get("index") != 0 for c in choices):
        raise ValueError("Ambiguous chat completion identity or choice")
    if len([c for c in choices if c.get("finish_reason") is not None]) != 1 or choices[-1].get("finish_reason") is None:
        raise ValueError("Incomplete chat completion")
    from nemo_gym.base_responses_api_model import _reconstruct_chat_sse
    return _reconstruct_chat_sse(events)


def _tools(response, dialect):
    if dialect == "responses":
        if response.get("status") != "completed":
            return []
        return [(item.get("call_id"),
                 f"{item['namespace']}__{item.get('name', '')}" if item.get("namespace") else item.get("name"),
                 item.get("arguments"))
                for item in response.get("output", []) if item.get("type") == "function_call"]
    choices = response.get("choices", [])
    if len(choices) != 1 or choices[0].get("index") != 0:
        raise ValueError("Process scoring requires one captured completion")
    choice = choices[0]
    if choice.get("finish_reason") not in {"stop", "tool_calls"}:
        return []
    tools = choice.get("message", {}).get("tool_calls")
    # Gym's typed Chat Completion serializes absent optional calls as null.
    # Final text answers carry no preview evidence, but must reach the harness.
    if tools is None:
        return []
    if not isinstance(tools, list):
        raise ValueError("Expected a list of tool calls or null")
    return [(tool.get("id"), tool.get("function", {}).get("name"), tool.get("function", {}).get("arguments"))
            for tool in tools]


@dataclass(frozen=True)
class PreviewCall:
    rollout_id: str
    response_id: str
    model_call_id: str
    tool_call_id: str
    tool_name: str
    arguments_json: str
    render_id: str


class PreviewCallObserver:
    def __init__(self, rollout_id, manifest_getter, *, tool_names, max_calls=4096):
        if (not re.fullmatch(r"[A-Za-z0-9_.-]{1,160}", rollout_id)
                or not tool_names or any(not isinstance(n, str) or not n for n in tool_names)
                or type(max_calls) is not int or max_calls <= 0):
            raise ValueError("Explicit rollout, preview tool names and positive capacity required")
        self.rollout_id, self.manifest_getter = rollout_id, manifest_getter
        self.tool_names, self.max_calls = frozenset(tool_names), max_calls
        self._calls, self._claimed, self._requests = {}, set(), {}
        self._lock = asyncio.Lock()

    async def __call__(self, path, content_type, raw):
        prefix = f"/ng-rollout/{self.rollout_id}/training-token-capture/v1/"
        if path not in {prefix + "responses", prefix + "chat/completions"}:
            raise InfrastructureError("Preview observer received a foreign rollout or route")
        dialect = "responses" if path.endswith("/responses") else "chat"
        response = _served_response(content_type, raw, dialect)
        candidates = []
        for call_id, name, arguments in _tools(response, dialect):
            if name not in self.tool_names:
                continue
            try:
                arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
                if not isinstance(arguments, dict):
                    continue
                canonical = _json(arguments)
            except (TypeError, ValueError):
                # Malformed model arguments remain a tool validation error,
                # not a transport failure or a fabricated process grade.
                continue
            if not isinstance(call_id, str) or not call_id:
                raise InfrastructureError("Preview tool lacks a served call identity")
            candidates.append((call_id, name, canonical))
        if not candidates:
            return
        response_id = response.get("id")
        if not isinstance(response_id, str) or not response_id:
            raise InfrastructureError("Preview response lacks its served envelope ID")
        async with self._lock:
            manifest = await self.manifest_getter(self.rollout_id)
            if manifest.rollout_id != self.rollout_id or manifest.failures:
                raise InfrastructureError("Preview capture ledger is invalid")
            records = [r for r in manifest.records if r.response_id == response_id]
            if len(records) != 1:
                raise InfrastructureError("Served response does not identify exactly one committed call")
            proposed = {}
            for call_id, name, arguments in candidates:
                identity = records[0].model_call_id, call_id
                render_id = hashlib.sha256(_json([self.rollout_id, *identity]).encode()).hexdigest()
                call = PreviewCall(self.rollout_id, response_id, identity[0], call_id,
                                   name, arguments, render_id)
                old = proposed.get(identity, self._calls.get(identity))
                if old is not None and old != call:
                    raise InfrastructureError("Captured preview identity changed its arguments")
                proposed[identity] = call
            if len(self._calls.keys() | proposed.keys()) > self.max_calls:
                raise InfrastructureError("Preview capture capacity exceeded")
            self._calls.update(proposed)

    async def claim(self, request_id, tool_name, arguments):
        """Called by the session-owned tool endpoint, never with agent call IDs.

        MCP request IDs provide idempotency only. Association comes from one
        unconsumed, exact name/arguments match in the trusted served responses.
        Parallel indistinguishable calls fail closed rather than using recency.
        """
        if type(request_id) not in (str, int) or (isinstance(request_id, str) and not request_id):
            raise ValueError("A tool transport request identity is required")
        signature = tool_name, _json(arguments)
        key = type(request_id).__name__, request_id
        async with self._lock:
            if key in self._requests:
                previous, call = self._requests[key]
                if previous != signature:
                    raise InfrastructureError("Tool retry changed its request")
                if any(identity not in self._claimed
                       and (tool_name is None or candidate.tool_name == tool_name)
                       and candidate.arguments_json == signature[1]
                       for identity, candidate in self._calls.items()):
                    # JSON-RPC clients may recycle IDs after a reconnect. A
                    # retry and a new identical tool call are then ambiguous;
                    # never silently reuse the older version's score.
                    raise InfrastructureError("Tool request ID conflicts with a new captured preview")
                return call
            # None is used only by the trusted preview endpoint to accept the
            # explicitly configured per-harness aliases of that one tool.
            matches = [(identity, call) for identity, call in self._calls.items()
                       if identity not in self._claimed
                       and (tool_name is None or call.tool_name == tool_name)
                       and call.arguments_json == signature[1]]
            if len(matches) != 1:
                raise InfrastructureError("Tool request has no unique captured preview call")
            identity, call = matches[0]
            self._claimed.add(identity)
            self._requests[key] = signature, call
            return call
