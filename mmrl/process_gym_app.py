"""Opt-in process evidence collection in real CodingResources sessions.

This entry point emits trusted stage metadata. Composition pairs it with the
optional native per-token return hooks. The normal app.py entry remains unchanged.
"""
from dataclasses import asdict
import hashlib
import inspect
import os
from pathlib import Path
import re

from fastapi import Request
from pydantic import Field, PrivateAttr, model_validator

from nemo_gym.global_config import get_first_server_config_dict
from nemo_gym.server_utils import SESSION_ID_KEY
from nemo_gym.token_id_capture.control_routes import RolloutControlClient

from .gym_app import CodingResourcesServer, Config, RunRequest, SeedResponse, VerifyRequest
from .judge_contract import JUDGE_INSTRUCTIONS, RUBRICS
from .process_observer import PreviewCallObserver
from .process_rewards import ProcessRewardConfig
from .process_stages import StageCollector
from .process_tool import PreviewSession, install_preview_route
from .reward import InfrastructureError
from .tls import managed_gateway


class ProcessConfig(Config):
    expose_tools_over_mcp: bool = True
    capture_gateway_port: int = Field(ge=1024, le=65535)
    verification_log_dir: str
    capture_control_token_env: str = "NEMO_GYM_TOKEN_CAPTURE_CONTROL_TOKEN"
    process_reward_config: dict = Field(default_factory=lambda: asdict(ProcessRewardConfig()))

    @model_validator(mode="after")
    def process_contract(self):
        if not self.expose_tools_over_mcp or not self.verification_log_dir:
            raise ValueError("Process evidence requires signed MCP tools and private audit storage")
        ProcessRewardConfig(**self.process_reward_config)
        return self


class ProcessSeedResponse(SeedResponse):
    mmrl_preview_url: str | None = None
    mmrl_preview_headers: dict | None = None


class ProcessCodingResourcesServer(CodingResourcesServer):
    config: ProcessConfig
    _capture_sessions: dict[str, str | None] = PrivateAttr(default_factory=dict)
    _control_client: object = PrivateAttr(default=None)

    def setup_webserver(self):
        from nemo_gym.mcp_auto_exposure import call_direct
        if "transport_request_id" not in inspect.signature(call_direct).parameters:
            raise RuntimeError("Apply the optional Gym process-tools patch before starting this entry point")
        app = super().setup_webserver()
        install_preview_route(app, lambda sid: getattr(self._sessions.get(sid), "preview", None))
        return app

    def _gateway_context(self):
        return managed_gateway(
            self.config.capture_gateway_port, self.config.host, self.server_client.global_config_dict,
            response_observer=self._observe_response,
            tool_upstream=f"http://{self.config.host}:{self.config.port}")

    async def _manifest(self, rollout_id):
        if self._control_client is None:
            policy = get_first_server_config_dict(self.server_client.global_config_dict, "policy_model")
            self._control_client = RolloutControlClient(
                f"http://{policy.host}:{policy.port}",
                auth_token=os.environ.get(self.config.capture_control_token_env, ""), request_timeout_s=30)
        return await self._control_client.manifest(rollout_id)

    async def _observe_response(self, path, content_type, raw):
        match = re.fullmatch(r"/ng-rollout/([A-Za-z0-9_.-]{1,160})/training-token-capture/v1/(?:responses|chat/completions)", path)
        if match is None:
            raise InfrastructureError("Unsupported process capture path")
        sid = self._capture_sessions.get(match[1])
        session = self._sessions.get(sid)
        if session is None:
            raise InfrastructureError("Process response has no live resource session")
        if session.preview is not None:
            await session.preview.observer(path, content_type, raw)

    async def seed_session(self, request: Request, body: RunRequest) -> ProcessSeedResponse:
        rollout = body.capture_rollout_id
        if not rollout or not re.fullmatch(r"[A-Za-z0-9_.-]{1,160}", rollout):
            raise ValueError("Process evidence requires a native capture rollout identity")
        if rollout in self._capture_sessions:
            raise InfrastructureError("Capture rollout already owns a resource session")
        self._capture_sessions[rollout] = None  # Reserve before awaiting provisioning.
        sid = None
        try:
            seed = await super().seed_session(request, body)
            sid = request.session["mm_session"]
            request.session[SESSION_ID_KEY] = sid
            self._capture_sessions[rollout] = sid
            session = self._sessions[sid]
            if session.route == "single":
                return ProcessSeedResponse(**seed.model_dump())
            token = os.environ.get("MMRL_CAPTURE_GATEWAY_TOKEN", "")
            if len(token) < 32:
                raise InfrastructureError("Preview gateway credentials unavailable")
            conditions = {
                "sandbox_image": self.config.sandbox_image,
                "judge_instructions_sha256": hashlib.sha256(JUDGE_INSTRUCTIONS.encode()).hexdigest(),
                "rubric_sha256": hashlib.sha256(RUBRICS[session.task.benchmark].encode()).hexdigest(),
                "renderer_sha256": hashlib.sha256(Path(__file__).with_name("render.py").read_bytes()).hexdigest(),
                "service_sha256": hashlib.sha256(Path(__file__).with_name("service.py").read_bytes()).hexdigest(),
            }
            collector = StageCollector(
                rollout_id=rollout, task=session.task, route=session.route,
                sandbox=self._sandbox, service=self._service, render_conditions=conditions,
                audit_directory=Path(self.config.verification_log_dir) / "process" / rollout)
            observer = PreviewCallObserver(rollout, self._manifest, tool_names={
                "mmrl__preview", "mcp__mmrl__preview", "mcp_mmrl_preview", "mmrl_preview"})
            session.preview = PreviewSession(observer, collector, session.handle, None)
            return ProcessSeedResponse(
                **seed.model_dump(), mmrl_preview_url=f"https://{self.config.host}:{self.config.capture_gateway_port}/mcp",
                mmrl_preview_headers={"Authorization": "Bearer " + token})
        except BaseException:
            self._capture_sessions.pop(rollout, None)
            if sid is not None:
                session = self._pop_session(sid)
                if session is not None and session.handle is not None:
                    await self._close(session.handle)
            raise

    def mcp_allowed_tools_for_session(self, seed_body):
        return [] if seed_body.get("route") == "single" else ["preview"]

    def _pop_session(self, session_id):
        session = super()._pop_session(session_id)
        if session is not None:
            self._capture_sessions.pop(session.capture_rollout_id, None)
        return session

    def _before_verify(self, session):
        if session.preview is not None and session.preview.unresolved_failures:
            raise InfrastructureError("Process verification has unresolved infrastructure failures")

    async def verify(self, request: Request, body: VerifyRequest):
        session = self._sessions.get(request.session.get("mm_session", ""))
        result = await super().verify(request, body)
        if session is not None and not result.mask_sample:
            # A single-turn row in a mixed run carries explicit final-only
            # returns. It gets no preview tool, process bonus or efficiency cost.
            coefficients = ProcessRewardConfig(**self.config.process_reward_config)
            stages = []
            if session.route == "single":
                coefficients = ProcessRewardConfig(progress_weight=0, token_cost_per_1000=0,
                    render_cost=0, duplicate_render_cost=0)
            elif session.preview is not None:
                stages = [asdict(stage) for stage in session.preview.collector.stages]
            else:
                raise InfrastructureError("Multi-turn process verification lost its preview session")
            result.model_extra["process_reward_payload"] = {
                "version": 1, "rollout_id": session.capture_rollout_id,
                "final_reward": result.reward, "stages": stages,
                "config": asdict(coefficients),
            }
        return result


if __name__ == "__main__":
    ProcessCodingResourcesServer.run_webserver()
