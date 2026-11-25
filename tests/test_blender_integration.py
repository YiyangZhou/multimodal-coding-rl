import asyncio
import json
from unittest.mock import AsyncMock

from PIL import Image
import pytest

from mmrl.blender_sandbox import ObjectEvidence, VIEWS, classify_receipt, contact_sheet
from mmrl.reward import InfrastructureError, Quality
from mmrl.service import CodingService
from mmrl.tasks import Task, gym_row


@pytest.mark.parametrize("status", ["ERR_EXEC", "ERR_NO_MESH"])
def test_upstream_zero_exit_cannot_turn_model_error_into_success(status):
    assert not classify_receipt({"status": status, "n_views_rendered": 0})


@pytest.mark.parametrize("receipt", [
    {"status": "OK", "n_views_rendered": 3, "n_meshes": 1},
    {"status": "OK", "n_views_rendered": 4, "n_meshes": 0},
    {"status": "OK", "n_views_rendered": 4, "n_meshes": True},
    {"status": "ERR_RENDER"}, {"status": "ERR_NOLOG"}, [],
])
def test_incomplete_or_unusable_views_are_not_quality_measurements(receipt):
    with pytest.raises(InfrastructureError):
        classify_receipt(receipt)


def test_contact_sheet_requires_all_canonical_views(tmp_path):
    for name, _ in VIEWS:
        Image.new("RGBA", (512, 512), "red").save(tmp_path / name)
    png, hashes = contact_sheet(tmp_path)
    assert png.startswith(b"\x89PNG")
    assert len(hashes) == 4
    (tmp_path / VIEWS[2][0]).unlink()
    with pytest.raises(InfrastructureError):
        contact_sheet(tmp_path)


def test_reference_code_is_private_for_both_text_and_image_inputs(tmp_path):
    reference = tmp_path / "reference.png"
    Image.new("RGB", (32, 32)).save(reference)
    for views in ([], [str(reference)] * 4):
        task = Task("3dcodebench", "shape", "Create the requested object", views,
                    ground_truth="PRIVATE_FACTORY_CODE")
        row = gym_row(task, "single")
        assert "PRIVATE_FACTORY_CODE" not in json.dumps(row["responses_create_params"])
        assert row["task"]["ground_truth"] == "PRIVATE_FACTORY_CODE"
        assert sum(item["type"] == "input_image" for item in row["responses_create_params"]["input"][-1]["content"]) == len(views)


def test_separate_reference_views_and_reward_contract():
    async def run():
        sandbox, judge = AsyncMock(), AsyncMock()
        sandbox.render_blender.side_effect = [ObjectEvidence(b"candidate", {}), ObjectEvidence(b"reference", {})]
        judge.grade.return_value = Quality(.8, .6)
        task = Task("3dcodebench", "shape", "A chair", ground_truth="factory()")
        result = await CodingService(sandbox, judge).score(task, "```python\nchair()\n```")
        assert result["reward"] == pytest.approx(.76)
        assert [c.args for c in sandbox.render_blender.await_args_list] == [("chair()",), ("factory()",)]
        assert judge.grade.await_args.args[2:4] == (b"candidate", [b"reference"])
    asyncio.run(run())


def test_candidate_failure_skips_api_and_reference_failure_is_excluded():
    async def run():
        sandbox, judge = AsyncMock(), AsyncMock()
        task = Task("3dcodebench", "shape", "A chair", ground_truth="factory()")
        service = CodingService(sandbox, judge)
        sandbox.render_blender.return_value = None
        result = await service.score(task, "chair()")
        assert result["reward"] == result["execution"] == 0
        judge.grade.assert_not_awaited()
        sandbox.render_blender.side_effect = [ObjectEvidence(b"candidate", {}), None]
        with pytest.raises(InfrastructureError):
            await service.score(task, "chair()")
        judge.grade.assert_not_awaited()
    asyncio.run(run())
