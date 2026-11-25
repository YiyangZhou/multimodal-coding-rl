import json
from pathlib import Path

from .artifacts import extract_answer
from .webmmu_patch import apply_webmmu_patch
from .web_document import compose_document
from .catalog import BENCHMARKS
from .reward import InfrastructureError


class CodingService:
    """One final-reward path for all four rollout routes."""

    def __init__(self, sandbox, judge):
        self.sandbox, self.judge = sandbox, judge

    async def score(self, task, answer, *, harness_artifact=False, audit=None):
        if audit is not None:
            audit.write("answer.txt", answer)
        original_document = None
        if task.benchmark == "webmmu_code_edit":
            try:
                original_document = compose_document(task.original_source)
            except ValueError as exc:
                raise InfrastructureError("Code Edit source lacks a supported complete document") from exc
        try:
            candidate = (
                answer
                if harness_artifact
                else extract_answer(answer, BENCHMARKS[task.benchmark].language)
            )
            if not candidate.strip():
                raise ValueError("Empty artifact")
            render_source = (
                compose_document(apply_webmmu_patch(task.original_source, candidate))
                if task.benchmark == "webmmu_code_edit"
                else candidate
            )
        except ValueError as exc:
            return {
                "reward": 0.0,
                "execution": 0.0,
                "failure_kind": "mm_coding:invalid_artifact",
                "failure_reason": str(exc),
            }
        if audit is not None:
            audit.write("render_source.txt", render_source)
        if task.benchmark == "vision2web":
            if not all(hasattr(self.judge, name) for name in ("advance", "check", "grade_visual")):
                raise InfrastructureError("Vision2Web requires the trusted workflow judge")
            return await self.sandbox.score_vision(task, candidate, self.judge, audit=audit)
        media = {}
        reference_media = {}
        if audit is not None and audit.capture_media:
            media = {"media_directory": audit.path / "candidate-media"}
            reference_media = {"media_directory": audit.path / "reference-media"}
        animation = None
        object_views = None
        if task.benchmark == "chartmimic":
            png = await self.sandbox.render_chartmimic(render_source)
        elif task.benchmark == "manimbench_v1":
            animation = await self.sandbox.render_manim(render_source, **media)
            png = animation.png if animation is not None else None
        elif task.benchmark == "3dcodebench":
            object_views = await self.sandbox.render_blender(render_source, **media)
            png = object_views.png if object_views is not None else None
        else:
            png = await self.sandbox.render(task.benchmark, render_source)
        if png is None:
            return {
                "reward": 0.0,
                "execution": 0.0,
                "failure_kind": "mm_coding:execution_failed",
            }
        if audit is not None:
            audit.write("render.png", png)
            if animation is not None:
                audit.write("animation.json", json.dumps(animation.metadata))
            if object_views is not None:
                audit.write("object_views.json", json.dumps(object_views.metadata))
        try:
            references = [Path(p).read_bytes() for p in (task.judge_references or task.references)]
        except OSError as exc:
            raise InfrastructureError(
                "Reference assets unavailable to the trusted verifier"
            ) from exc
        evidence = {}
        if task.benchmark == "3dcodebench":
            reference = await self.sandbox.render_blender(task.ground_truth, **reference_media)
            if reference is None:
                raise InfrastructureError("3D reference failed to render; no valid quality target")
            references = [reference.png]
            if audit is not None:
                audit.write("reference_render.png", reference.png)
                audit.write("reference_object_views.json", json.dumps(reference.metadata))
        if task.benchmark == "manimbench_v1":
            # Reference code is never staged in the candidate's verification box.
            reference = await self.sandbox.render_manim(task.ground_truth, **reference_media)
            if reference is None:
                raise InfrastructureError("Manim reference failed to render; no valid quality target")
            references = [reference.png]
            if audit is not None:
                audit.write("reference_render.png", reference.png)
                audit.write("reference_animation.json", json.dumps(reference.metadata))
        if task.benchmark == "webmmu_code_edit":
            original = await self.sandbox.render(task.benchmark, original_document)
            if original is None:
                raise InfrastructureError("Original page cannot render for Code Edit verification")
            evidence["original_render"] = original
            if audit is not None:
                audit.write("original_render.png", original)
            try:
                target_source = compose_document(apply_webmmu_patch(task.original_source, task.reference_patch))
            except ValueError:
                # Some dataset references are partial diffs; keep their textual
                # evidence without treating the unmodified screenshot as a target.
                target_source = None
            if target_source is not None:
                target = await self.sandbox.render(task.benchmark, target_source)
                if target is not None:
                    evidence["target_render"] = target
                    if audit is not None:
                        audit.write("target_render.png", target)
        quality = await self.judge.grade(task, candidate, png, references, **evidence)
        if audit is not None and quality.reason:
            audit.write("judge_reason.txt", quality.reason)
        return {
            "reward": quality.reward,
            "execution": 1.0,
            "task_fidelity": quality.task_fidelity,
            "visual_fidelity": quality.visual_fidelity,
        }
