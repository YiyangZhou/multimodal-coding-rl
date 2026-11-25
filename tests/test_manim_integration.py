import asyncio
from contextlib import asynccontextmanager
import hashlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from PIL import Image
import pytest

from mmrl.audit import VerificationAudit
from mmrl.manim_sandbox import AnimationEvidence, render_manim
from mmrl.reward import InfrastructureError, Quality
from mmrl.service import CodingService
from mmrl.tasks import Task, gym_row, route


def task():
    return Task("manimbench_v1", "example", "Animate a circle moving right",
                ground_truth="PRIVATE_REFERENCE_CODE")


def test_private_reference_never_enters_policy_input():
    row = gym_row(task(), "single")
    assert "PRIVATE_REFERENCE_CODE" not in json.dumps(row["responses_create_params"])
    assert row["task"]["ground_truth"] == "PRIVATE_REFERENCE_CODE"
    assert route(task(), "mixed") == "single"
    for harness in ("codex", "opencode", "hermes"):
        routed = gym_row(task(), harness)
        assert "PRIVATE_REFERENCE_CODE" not in json.dumps(routed["responses_create_params"])
        assert routed["task"]["ground_truth"] == "PRIVATE_REFERENCE_CODE"
    assert route(task(), "multi") in {"codex", "opencode", "hermes"}


def test_new_catalog_entry_does_not_change_legacy_import(tmp_path, monkeypatch):
    from mmrl.cli import main

    names = ("chart2code", "mmsvg_img2svg", "webmmu_code_edit", "webmmu_mockup2code",
             "image2struct_latex", "mmsvg_text2svg", "design2code_full")
    for name in names:
        folder = tmp_path / name
        folder.mkdir()
        context = {"benchmark_id": name, "sample_id": "example", "prompt": "Original task",
                   "target_image_paths": [] if name == "mmsvg_text2svg" else ["reference.png"],
                   "original_source": "original html", "ground_truth": "reference diff"}
        (folder / "train_context_ready.jsonl").write_text(json.dumps({"reward_model": context}) + "\n")
    output = tmp_path / "imported.jsonl"
    monkeypatch.setattr("sys.argv", ["mmrl", "import-legacy", "--root", str(tmp_path), "--output", str(output)])
    main()
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert {row["benchmark"] for row in rows} == set(names)


def test_export_rejects_modified_prepared_data(tmp_path):
    path = Path(__file__).parents[1] / "scripts/export_manim_gym.py"
    spec = importlib.util.spec_from_file_location("export_manim_gym", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    row = gym_row(task(), "single")["task"] | {"split": "train"}
    source = tmp_path / "train.jsonl"
    source.write_text(json.dumps(row) + "\n")
    (tmp_path / "preparation.json").write_text(json.dumps({
        "revision": "pinned", "files_sha256": {
            source.name: hashlib.sha256(source.read_bytes()).hexdigest()}}))
    output = tmp_path / "gym.jsonl"
    report = module.export(tmp_path, output, "train")
    assert report["rows"] == 1
    assert json.loads(output.read_text())["route"] == "single"
    source.write_text(source.read_text().replace("Animate", "Changed"))
    with pytest.raises(ValueError, match="changed after"):
        module.export(tmp_path, tmp_path / "bad.jsonl", "train")


def test_candidate_and_reference_are_separate_and_audited(tmp_path):
    async def run():
        sandbox, judge = AsyncMock(), AsyncMock()
        sandbox.render_manim.side_effect = [
            AnimationEvidence(b"candidate-png", {"frames": [0, 1]}),
            AnimationEvidence(b"reference-png", {"frames": [0, 1, 2]}),
        ]
        judge.grade.return_value = Quality(.8, .6)
        audit = VerificationAudit(tmp_path, task=task(), route="single",
                                  response_id="test", status="completed")
        result = await CodingService(sandbox, judge).score(
            task(), "```python\nCANDIDATE_CODE\n```", audit=audit)
        audit.finish(result)
        assert result["reward"] == pytest.approx(.76)
        assert [call.args for call in sandbox.render_manim.await_args_list] == [
            ("CANDIDATE_CODE",), ("PRIVATE_REFERENCE_CODE",)]
        assert judge.grade.await_args.args[2:4] == (b"candidate-png", [b"reference-png"])
        record = json.loads((audit.path / "record.json").read_text())
        assert record["complete"]
        assert {"animation.json", "reference_animation.json", "reference_render.png"} <= record["files"].keys()
    asyncio.run(run())


def test_execution_failure_skips_reference_and_api():
    async def run():
        sandbox, judge = AsyncMock(), AsyncMock()
        sandbox.render_manim.return_value = None
        result = await CodingService(sandbox, judge).score(task(), "bad_python()")
        assert result["execution"] == result["reward"] == 0
        assert sandbox.render_manim.await_count == 1
        judge.grade.assert_not_awaited()
    asyncio.run(run())


@pytest.mark.parametrize("fault", ["reference", "environment", "judge"])
def test_infrastructure_is_never_a_model_zero(fault):
    async def run():
        sandbox, judge = AsyncMock(), AsyncMock()
        valid = AnimationEvidence(b"png", {"frames": [0]})
        sandbox.render_manim.side_effect = [valid, valid]
        if fault == "reference":
            sandbox.render_manim.side_effect = [valid, None]
        elif fault == "environment":
            sandbox.render_manim.side_effect = InfrastructureError("environment unavailable")
        else:
            judge.grade.side_effect = InfrastructureError("budget exceeded")
        with pytest.raises(InfrastructureError):
            await CodingService(sandbox, judge).score(task(), "scene()")
    asyncio.run(run())


@pytest.mark.parametrize("return_code,status,expected", [
    (0, "ok", "ok"), (20, "execution_failed", "model_zero"),
    (0, "execution_failed", "infra"), (30, "infrastructure_error", "infra"),
    (137, "ok", "infra"), (0, "corrupt", "infra"),
])
def test_transport_status_contract_and_cleanup(return_code, status, expected):
    class Box:
        closed = False
        prepare = AsyncMock()
        upload = AsyncMock()

        @asynccontextmanager
        async def session(self):
            try:
                yield "handle"
            finally:
                self.closed = True

    async def run():
        box = Box()
        box.provider = AsyncMock()
        box.provider.exec.side_effect = [
            SimpleNamespace(return_code=return_code, error_type=None),
            SimpleNamespace(return_code=0, error_type=None),
            SimpleNamespace(return_code=0, error_type=None),
        ]
        async def download(handle, remote, local):
            if remote.endswith(".json"):
                local.write_text(json.dumps({"status": status, "frames": [{}], "scenes": [{}],
                    "preview": "/workspace/evidence/contact-sheet.png"}))
            else:
                buffer = io.BytesIO()
                Image.new("RGB", (2, 2)).save(buffer, format="PNG")
                local.write_bytes(buffer.getvalue())
        box.provider.download_file.side_effect = download
        if expected == "infra":
            with pytest.raises(InfrastructureError):
                await render_manim(box, "candidate")
        else:
            evidence = await render_manim(box, "candidate")
            assert (evidence is None) == (expected == "model_zero")
        assert box.closed
    asyncio.run(run())
