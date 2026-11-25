"""Ordered workflow verification; failed app actions never trigger self-repair."""
from .reward import InfrastructureError


def require_observation(value):
    if not isinstance(value, dict) or value.get("status") not in {"ok", "app_failure"}:
        raise InfrastructureError("Browser did not return a usable observation")
    if value.get("snapshot_truncated"):
        raise InfrastructureError("Browser state is truncated; do not infer a complete verdict")
    if value["status"] == "ok" and (not isinstance(value.get("snapshot"), str)
                                     or not isinstance(value.get("url"), str)):
        raise InfrastructureError("Browser observation is missing its page state")
    return value


async def verify_workflow(workflow, browser, controller, *, max_operations_per_action=8):
    """browser exposes command(dict), screenshot(fullpage), and stop()."""
    cases, visuals = [], []
    started = False
    try:
        state = require_observation(await browser.command({"op": "start", "width": workflow.width, "height": workflow.height}))
        started = True
        if state["status"] == "app_failure":
            return {"workflow": workflow.index, "execution": 0, "reason": state["reason"], "cases": [], "visuals": []}
        for node in workflow.nodes:
            if node.kind == "visual":
                png = await browser.screenshot(fullpage=node.payload["fullpage"])
                visuals.append({"position": node.index, "name": node.payload["name"],
                                "fullpage": node.payload["fullpage"], "png": png})
                continue
            history, failure = [], None
            for action in node.payload["actions"]:
                current = []
                for _ in range(max_operations_per_action + 1):
                    state = require_observation(await browser.command({"op": "snapshot"}))
                    if state["status"] == "app_failure":
                        failure = state["reason"]
                        break
                    decision = await controller.advance(action, state, current,
                                                        png=await browser.screenshot(fullpage=False))
                    if decision["impossible"]:
                        failure = decision["reason"]
                        break
                    if decision["done"]:
                        if not current:
                            raise InfrastructureError("Planner claimed action completion without any browser operation")
                        break
                    if len(current) >= max_operations_per_action:
                        raise InfrastructureError("Workflow planner exhausted its operation budget")
                    observed = require_observation(await browser.command(decision["action"]))
                    operation = {"action": decision["action"], "result": observed}
                    current.append(operation)
                    history.append(operation)
                    if observed["status"] == "app_failure":
                        failure = observed["reason"]
                        break
                if failure:
                    break
            if failure:
                verdict = {"passed": False, "reason": failure}
            else:
                state = require_observation(await browser.command({"op": "snapshot"}))
                if state["status"] == "app_failure":
                    verdict = {"passed": False, "reason": state["reason"]}
                else:
                    verdict = await controller.check(node.payload, state, history,
                                                     png=await browser.screenshot(fullpage=False))
            cases.append({"index": node.index, "test_case": node.payload, **verdict, "process": history})
        return {"workflow": workflow.index, "execution": 1, "cases": cases, "visuals": visuals}
    finally:
        if started:
            await browser.stop()
