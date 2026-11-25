import asyncio
import json
import sys
from unittest.mock import AsyncMock

from PIL import Image
import pytest

from mmrl.reward import InfrastructureError, Quality
from mmrl.sandbox import RemoteSandbox
from mmrl.service import CodingService
from mmrl.tasks import Task, gym_row


def test_compose_requires_chart_renderer_before_writing_configuration(tmp_path, monkeypatch):
    from mmrl import cli
    base = tmp_path / "base.json"
    base.write_text(json.dumps({"policy": {"generation": {
        "max_new_tokens": 32768, "vllm_cfg": {"max_model_len": 65536}}}}))
    data = tmp_path / "data.jsonl"
    data.write_text(json.dumps(gym_row(chart_task(tmp_path), "single")) + "\n")
    output = tmp_path / "composed"
    runtime = "registry.test/runtime@sha256:" + "a" * 64
    chart = "registry.test/chart@sha256:" + "b" * 64
    args = ["mmrl", "compose", "--base", str(base), "--data", str(data),
            "--image", runtime, "--gateway", "auto", "--output", str(output)]
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(ValueError, match="requires --chartmimic-image"):
        cli.main()
    assert not output.exists()
    monkeypatch.setattr(sys, "argv", args + ["--chartmimic-image", "chart:latest"])
    with pytest.raises(ValueError, match="SHA-256"):
        cli.main()
    assert not output.exists()
    monkeypatch.setattr(cli, "training_config", lambda *a, **kw: {"fixture": True})
    monkeypatch.setattr(sys, "argv", args + ["--chartmimic-image", chart])
    cli.main()
    cfg = json.loads((output / "gym.json").read_text())
    resources = cfg["mm_coding"]["resources_servers"]["mm_coding"]
    assert resources["chartmimic_image"] == chart
    assert resources["sandbox_image"] == runtime
    sandbox = RemoteSandbox(resources["sandbox_image"], provider=AsyncMock(),
                            chartmimic_image=resources["chartmimic_image"])
    assert sandbox.image == runtime and sandbox.chartmimic_image == chart


def chart_task(tmp_path):
    source, target = tmp_path / "source.png", tmp_path / "private-target.png"
    Image.new("RGB", (32, 32), "red").save(source)
    Image.new("RGB", (32, 32), "blue").save(target)
    return Task("chartmimic", "case", "Change the chart data as requested", [str(source)],
                category="customized", ground_truth="PRIVATE_REFERENCE_CODE",
                judge_references=[str(target)])


def test_only_input_image_is_visible_to_policy(tmp_path):
    task = chart_task(tmp_path)
    row = gym_row(task, "single")
    serialized = json.dumps(row["responses_create_params"])
    assert "PRIVATE_REFERENCE_CODE" not in serialized
    assert "private-target" not in serialized
    images = [x for x in row["responses_create_params"]["input"][-1]["content"] if x["type"] == "input_image"]
    from mmrl.tasks import image_uri
    assert [x["image_url"] for x in images] == [image_uri(task.references[0])]
    assert row["task"]["judge_references"] == task.judge_references


def test_customized_task_cannot_silently_use_source_as_target():
    with pytest.raises(ValueError, match="private target"):
        Task("chartmimic", "x", "Change data", ["source.png"], category="customized")


def test_judge_receives_target_not_model_input(tmp_path):
    async def run():
        task = chart_task(tmp_path)
        sandbox, judge = AsyncMock(), AsyncMock()
        sandbox.render_chartmimic.return_value = b"candidate"
        judge.grade.return_value = Quality(.8, .6)
        result = await CodingService(sandbox, judge).score(task, "```python\nprint('x')\n```")
        assert result["reward"] == pytest.approx(.76)
        assert judge.grade.await_args.args[3] == [(tmp_path / "private-target.png").read_bytes()]
        sandbox.render.assert_not_awaited()
    asyncio.run(run())


def test_missing_private_target_is_infrastructure_not_model_zero(tmp_path):
    async def run():
        task = chart_task(tmp_path)
        (tmp_path / "private-target.png").unlink()
        sandbox, judge = AsyncMock(), AsyncMock()
        sandbox.render_chartmimic.return_value = b"candidate"
        with pytest.raises(InfrastructureError):
            await CodingService(sandbox, judge).score(task, "print('x')")
        judge.grade.assert_not_awaited()
    asyncio.run(run())


def test_candidate_failure_skips_api(tmp_path):
    async def run():
        sandbox, judge = AsyncMock(), AsyncMock()
        sandbox.render_chartmimic.return_value = None
        result = await CodingService(sandbox, judge).score(chart_task(tmp_path), "raise ValueError()")
        assert result["reward"] == result["execution"] == 0
        judge.grade.assert_not_awaited()
    asyncio.run(run())


def test_no_fallback_to_incompatible_renderer():
    async def run():
        sandbox = RemoteSandbox("example.org/image@sha256:" + "a" * 64, provider=AsyncMock())
        with pytest.raises(InfrastructureError, match="Python 3.9"):
            await sandbox.render_chartmimic("x")
        sandbox.provider.create.assert_not_awaited()
    asyncio.run(run())
