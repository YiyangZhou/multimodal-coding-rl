import asyncio
import copy
import io
from unittest.mock import AsyncMock

from PIL import Image
import pytest

from mmrl.reward import InfrastructureError, Quality
from mmrl.vision_reward import score_results
from mmrl.vision_workflows import parse_workflows
from mmrl.vision_judge import VisionJudgeController


def fixture(functional=True):
    stream = io.BytesIO()
    Image.new("RGB", (16, 16), "white").save(stream, "PNG")
    png = stream.getvalue()
    cases = [{"objective": "Save", "actions": [], "validations": ["Saved"]},
             {"objective": "Delete", "actions": [], "validations": ["Deleted"]}]
    workflows = parse_workflows([
        {"content": cases if functional else [], "prototype": {"home": {"idx": 0}}},
        {"prototype": {"home": {"idx": 0}}},
    ])
    results = []
    for w in workflows:
        results.append({"workflow": w.index, "execution": 1,
                        "cases": [{"index": n.index, "test_case": n.payload,
                                   "passed": n.index == 0, "reason": "Observed UI"}
                                  for n in w.nodes if n.kind == "functional"],
                        "visuals": [{"position": n.index, **n.payload, "png": png}
                                    for n in w.nodes if n.kind == "visual"]})
    refs = {(w.index, "home"): png for w in workflows}
    judge = AsyncMock()
    judge.grade_visual.side_effect = [Quality(0.9, 0.8, "First"), Quality(0.7, 0.6, "Second")]
    return workflows, results, refs, judge


def test_every_case_and_repeated_visual_counts_in_training_reward():
    workflows, results, refs, judge = fixture()
    score = asyncio.run(score_results("Task", workflows, results, refs, judge))
    assert score["functional_pass_rate"] == 0.5
    assert score["task_fidelity"] == 0.5
    assert score["visual_fidelity"] == pytest.approx(0.7)
    assert score["reward"] == pytest.approx(0.68)
    assert len(score["visual_scores"]) == 2
    assert judge.grade_visual.await_count == 2


def test_static_task_does_not_claim_perfect_functionality():
    workflows, results, refs, judge = fixture(functional=False)
    score = asyncio.run(score_results("Task", workflows, results, refs, judge))
    assert score["functional_pass_rate"] is None
    assert score["task_fidelity_source"] == "visible_semantic_fidelity"
    assert score["task_fidelity"] == pytest.approx(0.8)
    assert score["reward"] == pytest.approx(0.8)


def test_application_failure_does_not_call_visual_api():
    workflows, results, refs, judge = fixture()
    results[0] = {"workflow": 0, "execution": 0, "reason": "App did not serve a page"}
    score = asyncio.run(score_results("Task", workflows, results, refs, judge))
    assert score["reward"] == 0 and score["quality"] is None
    judge.grade_visual.assert_not_called()


@pytest.mark.parametrize("damage", ["missing_workflow", "missing_case", "duplicate_case", "wrong_case",
                                    "missing_visual", "wrong_position", "wrong_fullpage", "no_png",
                                    "missing_reference", "invalid_reference", "nonboolean_pass"])
def test_missing_or_mismatched_evidence_is_not_scored(damage):
    workflows, results, refs, judge = fixture()
    if damage == "missing_workflow":
        results.pop()
    elif damage == "missing_case":
        results[0]["cases"].pop()
    elif damage == "duplicate_case":
        results[0]["cases"][1] = copy.deepcopy(results[0]["cases"][0])
    elif damage == "wrong_case":
        results = copy.deepcopy(results)
        results[0]["cases"][0]["test_case"]["validations"] = ["Different condition"]
    elif damage == "missing_visual":
        results[1]["visuals"].clear()
    elif damage == "wrong_position":
        results[1]["visuals"][0]["position"] = 1
    elif damage == "wrong_fullpage":
        results[1]["visuals"][0]["fullpage"] = False
    elif damage == "no_png":
        del results[1]["visuals"][0]["png"]
    elif damage == "missing_reference":
        del refs[(1, "home")]
    elif damage == "invalid_reference":
        refs[(1, "home")] = b"not an image"
    else:
        results[0]["cases"][0]["passed"] = 1
    with pytest.raises(InfrastructureError):
        asyncio.run(score_results("Task", workflows, results, refs, judge))
    judge.grade_visual.assert_not_called()


def test_second_visual_api_failure_never_returns_partial_reward():
    workflows, results, refs, judge = fixture()
    judge.grade_visual.side_effect = [Quality(1, 1, "First"), InfrastructureError("API unavailable")]
    with pytest.raises(InfrastructureError):
        asyncio.run(score_results("Task", workflows, results, refs, judge))


def test_visual_judge_preserves_four_field_contract_and_labels(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEYS", "test-only-no-network")
    judge = VisionJudgeController()
    judge.structured = AsyncMock(return_value={"quality": 0.7, "task_fidelity": 0.8,
                                               "visual_fidelity": 0.6, "reason": "Matched"})
    quality = asyncio.run(judge.grade_visual("Task", {"workflow": 0}, b"candidate", b"reference"))
    assert quality == Quality(0.8, 0.6, "Matched")
    kwargs = judge.structured.call_args.kwargs
    assert kwargs["png"] == b"candidate" and kwargs["reference"] == b"reference"
    assert "QUALITY ONLY" in kwargs["instructions"]
    judge.structured.return_value["quality"] = 1
    with pytest.raises(InfrastructureError):
        asyncio.run(judge.grade_visual("Task", {}, b"candidate", b"reference"))
