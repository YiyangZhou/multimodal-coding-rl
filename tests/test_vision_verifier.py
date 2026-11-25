import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from mmrl.reward import InfrastructureError
from mmrl.vision_browser import validate_action
from mmrl.vision_judge import parse_decision
from mmrl.vision_verifier import verify_workflow
from mmrl.vision_workflows import parse_workflows


@pytest.mark.parametrize("action", [
    {"op": "shell", "command": "cat /etc/passwd"}, {"op": "evaluate", "js": "1+1"},
    {"op": "click", "locator": {"kind": "css", "name": "script", "role": ""}},
    {"op": "press", "key": "Control+Shift+I"}, {"op": "reload", "url": "https://outside.test"},
])
def test_verifier_cannot_request_shell_source_or_javascript(action):
    with pytest.raises(ValueError):
        validate_action(action)
    with pytest.raises(InfrastructureError):
        parse_decision({"done": False, "impossible": False, "reason": "test", "action_json": json.dumps(action)})


def test_terminal_decision_cannot_hide_an_extra_action():
    with pytest.raises(InfrastructureError):
        parse_decision({"done": True, "impossible": False, "reason": "done", "action_json": '{"op":"reload"}'})


def workflow():
    return parse_workflows([{"content": [
        {"objective": "Click", "actions": ["Click Submit"], "validations": ["Saved"]},
        {"objective": "Check state", "actions": [], "validations": ["Still on page"]},
    ], "prototype": {"after_first": {"idx": 1}}}])[0]


def test_failed_action_is_not_retried_and_next_case_keeps_state():
    async def run():
        browser, judge = AsyncMock(), AsyncMock()
        operations = []
        async def command(action):
            operations.append(action["op"])
            if action["op"] == "click":
                return {"status": "app_failure", "reason": "action_timeout"}
            return {"status": "ok", "url": "http://localhost:3000", "snapshot": "Page"}
        browser.command.side_effect = command
        browser.screenshot.return_value = b"png"
        judge.advance.return_value = {"done": False, "impossible": False,
            "action": {"op": "click", "locator": {"kind": "role", "name": "Submit", "role": "button"}}}
        judge.check.return_value = {"passed": True, "reason": "Page remains visible"}
        result = await verify_workflow(workflow(), browser, judge)
        assert [c["passed"] for c in result["cases"]] == [False, True]
        assert operations.count("start") == operations.count("click") == 1
        assert [v["name"] for v in result["visuals"]] == ["after_first"]
        assert result["execution"] == 1
        browser.stop.assert_awaited_once()
    asyncio.run(run())


def test_api_failure_aborts_measurement_without_model_zero():
    async def run():
        browser, judge = AsyncMock(), AsyncMock()
        browser.command.return_value = {"status": "ok", "url": "http://localhost:3000", "snapshot": "Page"}
        browser.screenshot.return_value = b"png"
        judge.advance.side_effect = InfrastructureError("API unavailable")
        with pytest.raises(InfrastructureError):
            await verify_workflow(workflow(), browser, judge)
        browser.stop.assert_awaited_once()
    asyncio.run(run())


def test_claimed_action_without_execution_is_rejected():
    async def run():
        browser, judge = AsyncMock(), AsyncMock()
        browser.command.return_value = {"status": "ok", "url": "http://localhost:3000", "snapshot": "Page"}
        browser.screenshot.return_value = b"png"
        judge.advance.return_value = {"done": True, "impossible": False, "reason": "done"}
        with pytest.raises(InfrastructureError, match="without any browser operation"):
            await verify_workflow(workflow(), browser, judge)
    asyncio.run(run())
