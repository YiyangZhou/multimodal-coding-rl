"""Session-owned Gym preview tool; private grades never enter tool feedback."""
from dataclasses import dataclass, field
from typing import Literal

from fastapi import HTTPException, Request
from pydantic import BaseModel, ConfigDict

from .reward import InfrastructureError


class PreviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    view: Literal["default"] = "default"


@dataclass
class PreviewSession:
    observer: object
    collector: object
    handle: object
    wire_tool_name: str | None
    unresolved_failures: set = field(default_factory=set)

    async def invoke(self, request_id, arguments):
        key = type(request_id).__name__, request_id
        try:
            result = await self._invoke(request_id, arguments)
        except InfrastructureError:
            self.unresolved_failures.add(key)
            raise
        self.unresolved_failures.discard(key)
        return result

    async def _invoke(self, request_id, arguments):
        call = await self.observer.claim(request_id, self.wire_tool_name, arguments)
        await self.collector.capture_from_sandbox(
            self.handle, model_call_id=call.model_call_id, render_id=call.render_id)
        preview = self.collector.preview_file(call.render_id)
        if preview is None:
            return {"executed": False, "message": "Rendering failed. Repair the saved code and preview again."}
        destination = f"/workspace/.mmrl-preview-{call.render_id}.png"
        try:
            await self.collector.sandbox.provider.upload_file(self.handle, preview, destination)
        except Exception as exc:
            raise InfrastructureError("Verified preview delivery failed") from exc
        return {"executed": True, "preview_path": destination,
                "message": "Inspect this image with your image tool before deciding whether to revise."}


def install_preview_route(app, session_lookup):
    """Register a typed route for Gym's native, signed-session MCP exposure.

    session_lookup is a trusted resources-server callback. Direct HTTP requests
    cannot set request.state.mcp_request_id; the optional native MCP patch sets
    it from the SDK transport context, never from tool arguments or headers.
    """
    from nemo_gym.server_utils import SESSION_ID_KEY

    async def preview(request: Request, body: PreviewRequest):
        request_id = getattr(request.state, "mcp_request_id", None)
        session_id = request.session.get(SESSION_ID_KEY)
        if request_id is None or not session_id:
            raise HTTPException(403, "Use the session's authenticated Gym MCP tool")
        session = session_lookup(session_id)
        if session is None:
            raise HTTPException(403, "Preview session is unavailable")
        try:
            return await session.invoke(request_id, body.model_dump(exclude_unset=True))
        except InfrastructureError:
            # The session owner must retain this failure for final eligibility;
            # no judge/registry credentials or private grade details go to CLI.
            raise HTTPException(503, "Preview infrastructure unavailable") from None

    app.post("/preview", description="Independently render saved task code and return an image to inspect.")(preview)
