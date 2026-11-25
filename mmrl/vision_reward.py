"""Vision2Web training reward, distinct from the upstream leaderboard metrics.

All functional cases and visual checkpoints count, including repeated prototypes
in different workflows. Incomplete measurements raise rather than score zero.
"""
import hashlib

from .reward import InfrastructureError, Quality, image_data_uri
from .vision_verifier import verify_workflow


def validate_image(value):
    if not isinstance(value, bytes) or not value:
        raise InfrastructureError("Missing image evidence")
    image_data_uri(value)


def validate_coverage(workflows, results):
    if not workflows or len(workflows) != len(results):
        raise InfrastructureError("Incomplete Vision2Web workflow coverage")
    if [w.index for w in workflows] != list(range(len(workflows))):
        raise InfrastructureError("Unexpected Vision2Web workflow ordering")
    execution = 1
    for workflow, result in zip(workflows, results, strict=True):
        if (not isinstance(result, dict) or result.get("workflow") != workflow.index
                or type(result.get("execution")) is not int or result["execution"] not in (0, 1)):
            raise InfrastructureError("Invalid Vision2Web execution receipt")
        if not result["execution"]:
            if not isinstance(result.get("reason"), str) or not result["reason"]:
                raise InfrastructureError("Missing application failure reason")
            execution = 0
            continue
        expected_cases = [n for n in workflow.nodes if n.kind == "functional"]
        expected_visuals = [n for n in workflow.nodes if n.kind == "visual"]
        cases, visuals = result.get("cases"), result.get("visuals")
        if not isinstance(cases, list) or not isinstance(visuals, list):
            raise InfrastructureError("Missing workflow evidence")
        if len(cases) != len(expected_cases) or len(visuals) != len(expected_visuals):
            raise InfrastructureError("Incomplete workflow node coverage")
        for node, case in zip(expected_cases, cases, strict=True):
            if (not isinstance(case, dict) or case.get("index") != node.index or case.get("test_case") != node.payload
                    or type(case.get("passed")) is not bool or not isinstance(case.get("reason"), str)
                    or not case["reason"]):
                raise InfrastructureError("Functional receipt does not match the prescribed case")
        for node, visual in zip(expected_visuals, visuals, strict=True):
            if (not isinstance(visual, dict) or visual.get("position") != node.index or visual.get("name") != node.payload["name"]
                    or visual.get("fullpage") is not node.payload["fullpage"]):
                raise InfrastructureError("Visual receipt does not match the prescribed checkpoint")
            validate_image(visual.get("png"))
    return execution


async def score_results(prompt, workflows, results, references, judge):
    """references maps (workflow index, prototype name) to trusted image bytes.

    Task fidelity = case pass rate when cases exist, otherwise mean visual-judge
    semantic fidelity. Visual fidelity = mean over all visual checkpoints.
    No functional cases is reported as None, never a fabricated 100% pass rate.
    """
    expected_refs = {(w.index, n.payload["name"]) for w in workflows
                     for n in w.nodes if n.kind == "visual"}
    if not expected_refs or not isinstance(references, dict) or set(references) != expected_refs:
        raise InfrastructureError("Missing or unexpected reference prototypes")
    for reference in references.values():
        validate_image(reference)
    execution = validate_coverage(workflows, results)
    if not execution:
        return {"execution": 0, "reward": 0.0, "quality": None,
                "functional_pass_rate": None, "visual_scores": []}
    cases = [case for result in results for case in result["cases"]]
    scores = []
    for workflow, result in zip(workflows, results, strict=True):
        for visual in result["visuals"]:
            reference = references[(workflow.index, visual["name"])]
            checkpoint = {"workflow": workflow.index, "position": visual["position"],
                          "prototype": visual["name"], "fullpage": visual["fullpage"]}
            quality = await judge.grade_visual(prompt, checkpoint, visual["png"], reference)
            if not isinstance(quality, Quality):
                raise InfrastructureError("Visual judge did not return validated quality")
            scores.append({**checkpoint, "task_fidelity": quality.task_fidelity,
                           "visual_fidelity": quality.visual_fidelity, "reason": quality.reason,
                           "candidate_sha256": hashlib.sha256(visual["png"]).hexdigest(),
                           "reference_sha256": hashlib.sha256(reference).hexdigest()})
    passed = sum(case["passed"] for case in cases)
    functional = passed / len(cases) if cases else None
    semantic = functional if cases else sum(s["task_fidelity"] for s in scores) / len(scores)
    visual = sum(s["visual_fidelity"] for s in scores) / len(scores)
    quality = Quality(semantic, visual, "Complete workflow training aggregation")
    return {"execution": 1, "reward": quality.reward,
            "quality": (semantic + visual) / 2, "task_fidelity": semantic, "visual_fidelity": visual,
            "task_fidelity_source": "functional_case_pass_rate" if cases else "visible_semantic_fidelity",
            "functional_pass_rate": functional, "functional_cases": len(cases), "functional_passed": passed,
            "visual_scores": scores}


async def evaluate_workflows(prompt, workflows, browser, judge, references):
    """Run sequentially on one deployed app; keep database state across workflows."""
    results = []
    for workflow in workflows:
        results.append(await verify_workflow(workflow, browser, judge))
    return await score_results(prompt, workflows, results, references, judge), results
