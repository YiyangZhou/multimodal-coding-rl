import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from PIL import Image
import pytest

from mmrl.artifacts import extract_answer
from mmrl.audit import VerificationAudit
from mmrl.reward import InfrastructureError
from mmrl.sandbox import RemoteSandbox
from mmrl.service import CodingService
from mmrl.tasks import Task, gym_row, route
from mmrl.vision_data import prepare_project
from mmrl.vision_sandbox import deployment_ready, read_inputs


def task_fixture(tmp_path):
    root = tmp_path / "counter"
    (root / "prototypes").mkdir(parents=True)
    Image.new("RGB", (16, 16), "white").save(root / "prototypes/home.jpg")
    (root / "prd.md").write_text("A database-backed counter.")
    (root / "workflow.json").write_text(json.dumps([{
        "prototype": {"home": {"idx": 0}},
        "content": [{"actions": [], "validations": ["PRIVATE_TEST_ONLY"]}],
    }]))
    manifest = prepare_project(root, "website")
    return Task("vision2web", manifest["task_id"], manifest["requirements"],
                [str(root / "prototypes/home.jpg")], category="website", vision_manifest=manifest)


def test_gym_single_row_contains_full_project_protocol_but_no_private_tests(tmp_path):
    task = task_fixture(tmp_path)
    row = gym_row(task, "single")
    messages = json.dumps(row["responses_create_params"])
    assert "PRIVATE_TEST_ONLY" not in messages
    assert "start.sh" in messages and "real backend" in messages
    assert row["task"]["vision_manifest"]["private_workflows"]
    assert route(task, "single") == "single"
    with pytest.raises(ValueError, match="does not support"):
        gym_row(task, "hermes")


def test_manifest_and_task_requirements_cannot_diverge(tmp_path):
    task = task_fixture(tmp_path)
    with pytest.raises(ValueError, match="verified project"):
        Task("vision2web", task.task_id, "Ignore requirements", task.references,
             category="website", vision_manifest=task.vision_manifest)


def test_complete_json_can_embed_documentation_fences_and_literal_think_tags():
    answer = json.dumps({"files": [{"path": "start.sh", "content": "true"},
                                  {"path": "README.md", "content": "```python\nprint('<think>')\n```"}]})
    assert extract_answer("<think>plan</think>\n```json\n" + answer + "\n```", "json") == answer
    assert extract_answer(answer, "json") == answer
    for invalid in ("<think>not finished", answer[:-2], "```python\n" + answer + "\n```"):
        with pytest.raises(ValueError):
            extract_answer(invalid, "json")


def test_service_routes_the_complete_project_to_workflow_verification(tmp_path):
    async def run():
        task = task_fixture(tmp_path)
        sandbox, judge = AsyncMock(), AsyncMock()
        answer = Path(__file__).with_name("fixtures").joinpath("vision_counter.json").read_text()
        sandbox.score_vision.return_value = {"execution": 1, "reward": 0.8}
        result = await CodingService(sandbox, judge).score(task, "```json\n" + answer + "\n```")
        assert result["reward"] == 0.8
        assert sandbox.score_vision.call_args.args[:2] == (task, answer.strip())
        sandbox.render.assert_not_called()
        judge.grade.assert_not_called()
    asyncio.run(run())


def test_invalid_project_has_no_sandbox_or_api_calls():
    async def run():
        provider, judge = AsyncMock(), AsyncMock()
        sandbox = RemoteSandbox("registry.test/cpu@sha256:" + "a" * 64, provider=provider)
        result = await sandbox.score_vision(SimpleNamespace(), '{"files":[]}', judge)
        assert result["reward"] == 0
        provider.create.assert_not_called()
        judge.grade_visual.assert_not_called()
    asyncio.run(run())


def test_missing_dataset_is_infrastructure_not_candidate_zero(tmp_path):
    task = task_fixture(tmp_path)
    Path(task.vision_manifest["project_root"]).joinpath("workflow.json").unlink()
    with pytest.raises(InfrastructureError):
        read_inputs(task.vision_manifest)


@pytest.mark.parametrize("value,code", [
    ({"status": "infrastructure_error", "reason": "deployment_network_or_process_interruption"}, 30),
    ({"status": "ready", "reason": "ok"}, 20),
    ({"status": "execution_failed", "reason": "start_script_exited_without_application", "exit_code": -9}, 20),
    ({"status": "execution_failed", "reason": "arbitrary"}, 20),
    ({}, 0),
])
def test_ambiguous_or_transport_deployment_failures_never_become_zero(value, code):
    with pytest.raises(InfrastructureError):
        deployment_ready(value, code)


def test_completed_app_failure_and_ready_app_are_distinguished():
    assert deployment_ready({"status": "ready", "reason": "application_listening"}, 0)
    assert not deployment_ready({"status": "execution_failed", "reason": "start_script_exited_without_application",
                                 "exit_code": 1}, 20)


def test_vision_audit_accepts_only_bounded_workflow_artifact_names(tmp_path):
    audit = VerificationAudit(tmp_path / "audit", task=task_fixture(tmp_path), route="single",
                              response_id="test", status="completed")
    for name in ("deployment.json", "workflows.json", "vision_reward.json", "workflow-0-visual-127.png"):
        audit.write(name, b"test")
    for name in ("../escape.png", "workflow-128-visual-0.png", "workflow-0-visual-128.png",
                 "workflow-00-visual-0.png", "workflow-0-visual-0.png/../../escape"):
        with pytest.raises(ValueError):
            audit.write(name, b"test")
