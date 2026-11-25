"""Native NeMo Gym resources server; lifecycle shared with HarnessAgent."""

import asyncio
import logging
import secrets
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from time import monotonic

from fastapi import Request
from pydantic import ConfigDict, Field, PrivateAttr, model_validator

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)

from mmrl.audit import VerificationAudit
from mmrl.catalog import BENCHMARKS
from mmrl.reward import InfrastructureError
from mmrl.vision_judge import VisionJudgeController
from mmrl.sandbox import RemoteSandbox
from mmrl.service import CodingService
from mmrl.tasks import Task, reference_paths

LOG = logging.getLogger(__name__)


class Config(BaseResourcesServerConfig):
    sandbox_image: str
    chartmimic_image: str | None = None
    sandbox_memory_mib: int = Field(default=16384, ge=1024, strict=True)
    judge_concurrency: int = Field(default=8, ge=1)
    capture_gateway_port: int | None = Field(default=None, ge=1024, le=65535)
    verification_log_dir: str | None = None
    media_capture_task_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def one_session_owner(self):
        if self.num_workers not in (None, 1):
            raise ValueError(
                "Use one worker per resources service; scale with Gym shards"
            )
        return self


class RunRequest(BaseRunRequest):
    task: dict
    route: str


class VerifyRequest(BaseVerifyRequest):
    task: dict
    route: str


class VerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    instance_config: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def mirror_training_mask(self):
        """Keep the pinned V2 consumer aligned with Gym's canonical mask."""
        self.instance_config = {**self.instance_config, "mask_sample": self.mask_sample}
        return self


class SeedResponse(BaseSeedSessionResponse):
    sandbox_descriptor: dict | None = None


@dataclass
class Session:
    task: Task
    route: str
    handle: object | None
    created: float
    capture_rollout_id: str | None = None
    preview: object | None = None


class CodingResourcesServer(SimpleResourcesServer):
    config: Config
    _sessions: dict[str, Session] = PrivateAttr(default_factory=dict)
    _sandbox: object = PrivateAttr()
    _service: object = PrivateAttr()

    def model_post_init(self, __context):
        self._sandbox = RemoteSandbox(
            self.config.sandbox_image, memory_mib=self.config.sandbox_memory_mib,
            chartmimic_image=self.config.chartmimic_image,
        )
        self._service = CodingService(
            self._sandbox, VisionJudgeController(self.config.judge_concurrency)
        )

    async def _close(self, handle):
        try:
            await self._sandbox.provider.close(handle)
        except Exception as exc:
            # Cleanup must not replace a measured reward or kill all session reaping.
            # The remote TTL remains the final reclamation boundary.
            LOG.warning(
                "Sandbox cleanup failed (%s); remote TTL still applies",
                type(exc).__name__,
            )

    def _pop_session(self, session_id):
        return self._sessions.pop(session_id, None)

    def _before_verify(self, session):
        """Extension point for session-level eligibility checks."""

    def _gateway_context(self):
        from mmrl.tls import managed_gateway
        return managed_gateway(self.config.capture_gateway_port, self.config.host,
                               self.server_client.global_config_dict)

    def setup_webserver(self):
        app = super().setup_webserver()

        @asynccontextmanager
        async def lifespan(app):
            async def reap():
                while True:
                    await asyncio.sleep(60)
                    for key, session in list(self._sessions.items()):
                        if monotonic() - session.created > 12000:
                            self._pop_session(key)
                            if session.handle is not None:
                                await self._close(session.handle)

            async with AsyncExitStack() as stack:
                await stack.enter_async_context(self._gateway_context())
                task = asyncio.create_task(reap())
                try:
                    yield
                finally:
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                    try:
                        for key in list(self._sessions):
                            session = self._pop_session(key)
                            if session.handle is not None:
                                await self._close(session.handle)
                    finally:
                        self._sessions.clear()
                        await self._sandbox.provider.aclose()

        app.router.lifespan_context = lifespan
        return app

    async def seed_session(self, request: Request, body: RunRequest) -> SeedResponse:
        task = Task(**body.task)
        if body.route not in {"single", "opencode", "codex", "hermes"}:
            raise ValueError("Unknown rollout route")
        if body.route not in BENCHMARKS[task.benchmark].routes:
            raise ValueError("This benchmark does not support the requested rollout route")
        handle = None
        descriptor = None
        if body.route != "single":
            handle = await self._sandbox.create()
            try:
                await self._sandbox.prepare(handle, task.original_source)
                from .preview_install import install_extended_preview
                await install_extended_preview(self._sandbox, handle, task.benchmark)
                for reference, destination in zip(task.references, reference_paths(task.references)):
                    path = Path(reference)
                    if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
                        raise ValueError("Unsupported reference image format")
                    await self._sandbox.provider.upload_file(
                        handle, path, destination
                    )
                descriptor = await self._sandbox.provider.serialize_handle(handle)
                descriptor["workdir"] = "/workspace"
            except BaseException:
                await self._close(handle)
                raise
        session_id = secrets.token_urlsafe(24)
        self._sessions[session_id] = Session(task, body.route, handle, monotonic(), body.capture_rollout_id)
        request.session["mm_session"] = session_id
        return SeedResponse(sandbox_descriptor=descriptor)

    async def verify(self, request: Request, body: VerifyRequest) -> VerifyResponse:
        session = self._pop_session(request.session.get("mm_session", ""))
        if session is None:
            raise RuntimeError("Missing Gym session; no valid reward can be produced")
        audit = None
        result = None
        try:
            if body.capture_rollout_id != session.capture_rollout_id:
                raise RuntimeError("Rollout identity changed during verification")
            if body.route != session.route or Task(**body.task) != session.task:
                raise RuntimeError("Task changed during rollout")
            if self.config.verification_log_dir is not None:
                audit = VerificationAudit(
                    self.config.verification_log_dir,
                    task=session.task,
                    capture_media=session.task.task_id in self.config.media_capture_task_ids,
                    route=session.route,
                    response_id=body.response.id,
                    rollout_id=session.capture_rollout_id,
                    status=body.response.status,
                    failure_code=getattr(body.response.error, "message", None),
                    exit_diagnostics=(body.response.metadata or {}).get("harness_exit_diagnostics"),
                    terminal_metadata=body.response.metadata,
                )
            self._before_verify(session)
            if (
                body.response.status == "failed"
                or body.response.error is not None
                or (body.response.metadata or {}).get("failed") == "true"
                or (body.response.metadata or {}).get("interrupted") == "true"
            ):
                raise InfrastructureError(
                    "Harness or model request failed; no valid task score"
                )
            messages = [
                o
                for o in body.response.output
                if o.type == "message" and o.role == "assistant"
            ]
            final_text = (
                "".join(c.text for c in messages[-1].content if c.type == "output_text")
                if messages
                else ""
            )
            if audit is not None:
                audit.write("final_message.txt", final_text)
            if (
                body.response.status not in (None, "completed")
                or not final_text.strip()
            ):
                result = VerifyResponse(
                    **body.model_dump(),
                    reward=0.0,
                    execution=0.0,
                    failure_kind="mm_coding:incomplete",
                    # Receipt finalization must preserve exclusion even when
                    # the output budget is below the total context budget.
                    mask_sample=body.response.status not in (None, "completed"),
                )
                return result
            if session.handle is not None:
                try:
                    answer = await self._sandbox.read_artifact(
                        session.handle, BENCHMARKS[session.task.benchmark].artifact
                    )
                except ValueError as exc:
                    result = VerifyResponse(
                        **body.model_dump(),
                        reward=0.0,
                        execution=0.0,
                        failure_kind="mm_coding:invalid_artifact",
                        failure_reason=str(exc),
                    )
                    return result
            else:
                answer = final_text
            score = await self._service.score(
                session.task,
                answer,
                harness_artifact=session.handle is not None,
                audit=audit,
            )
            result = VerifyResponse(**body.model_dump(), **score)
            return result
        except InfrastructureError as exc:
            result = VerifyResponse(
                **body.model_dump(),
                reward=0.0,
                mask_sample=True,
                failure_kind="mm_coding:infrastructure",
                failure_reason=str(exc),
            )
            return result
        except OSError as exc:
            # Evidence storage is infrastructure, never a model execution zero.
            raise InfrastructureError(
                "Verification evidence could not be saved"
            ) from exc
        finally:
            try:
                if audit is not None and result is not None:
                    try:
                        audit.finish(result.model_dump())
                    except OSError as exc:
                        raise InfrastructureError(
                            "Verification evidence could not be finalized"
                        ) from exc
            finally:
                if session.handle is not None:
                    await self._close(session.handle)


if __name__ == "__main__":
    CodingResourcesServer.run_webserver()
