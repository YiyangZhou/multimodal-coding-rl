"""Exercise runner-log -> response JSON -> real resource HTTP audit boundaries."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import responses_api_agents.harness_agent.app as harness
import mmrl.audit as audit
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient

ROOT = Path(__file__).resolve().parents[1]

EXPECTED = {"returncode": 1, "http_statuses": [502], "markers": ["stream_disconnected"]}
RAW = json.dumps(EXPECTED | {"raw_stderr": "dummy-private-token", "url": "https://private.invalid"})


async def main():
    for module in (harness, audit):
        for invalid in (None, "{", "[]", '"text"', "x" * 9000, '{"returncode":true}', '{"returncode":256}', '{"returncode":1,"markers":{}}'):
            assert module.safe_exit_diagnostics(invalid) is None
        assert module.safe_exit_diagnostics(RAW) == EXPECTED
        assert module.safe_exit_diagnostics(json.dumps({"returncode": 1, "http_statuses": [False, 502, "secret", 700], "markers": ["secret", {}, "timeout"]})) == {"returncode":1,"http_statuses":[502],"markers":["timeout"]}

    config = harness.HarnessAgentConfig(
        host="127.0.0.1", port=18000, name="diagnostic-test", entrypoint="",
        resources_server={"type":"resources_servers","name":"rs"},
        model_server={"type":"responses_api_models","name":"model"},
        agent="codex", sandbox_provider={"opensandbox": {}},
    )
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {}
    provider = MagicMock()
    with patch.object(harness, "create_provider", return_value=provider), patch.object(harness.HarnessAgent, "_build_gym_tar", return_value=None):
        agent = harness.HarnessAgent(config=config, server_client=client)
    response = NeMoGymResponse(
        id="resp-test", created_at=1.0, model="policy", object="response", status="failed",
        parallel_tool_calls=True, tool_choice="auto", tools=[], output=[],
        error={"code":"server_error","message":"codex_nonzero_exit"}, metadata={"failed":"true"},
    )
    tested = 0
    for status, logs, expected in (
        ("failed", "WARNING codex exit diagnostics: " + RAW + "\nRUNNER_DONE", EXPECTED),
        ("failed", "WARNING codex exit diagnostics: {bad\nRUNNER_DONE", None),
        ("completed", "WARNING codex exit diagnostics: " + RAW + "\nRUNNER_DONE", None),
    ):
        returned = response.model_copy(deep=True)
        returned.status = status
        provider.exec = AsyncMock(side_effect=[SimpleNamespace(return_code=0, stderr=""), SimpleNamespace(stdout=logs, stderr="")])
        with patch.object(harness.HarnessAgent, "_runner", return_value=("unused", {}, "unused")), patch.object(harness.HarnessAgent, "_sandbox_model_url", return_value="https://model.invalid"), patch.object(harness.HarnessAgent, "_provision_box", new=AsyncMock(return_value=SimpleNamespace(provider_name="opensandbox"))), patch.object(harness.HarnessAgent, "_download_json", new=AsyncMock(return_value=returned.model_dump(mode="json"))), patch.object(harness.HarnessAgent, "_close_box", new=AsyncMock()) as close:
            result = await agent.responses(SimpleNamespace(), NeMoGymResponseCreateParamsNonStreaming(input="test"))
        close.assert_awaited_once()
        wire = result.model_dump_json()
        assert "dummy-private-token" not in wire and "private.invalid" not in wire
        roundtrip = NeMoGymResponse.model_validate_json(wire)
        raw = (roundtrip.metadata or {}).get("harness_exit_diagnostics")
        assert (json.loads(raw) if raw else None) == expected
        if raw:
            assert len(raw) <= 512
        assert result.status == status
        tested += 1

    # Reuse the actual ASGI /run -> seed -> verify audit/identity test, injecting
    # metadata only on Codex's failed response. No sandbox/model is contacted.
    source = (ROOT / "tests/check_audit_identity.py").read_text()
    source = source.replace('returned.status="failed"', 'returned.status="failed"\n                            if route == "codex":\n                                returned.metadata = {"harness_exit_diagnostics": ' + repr(RAW) + '}')
    needle = '            assert len(records)==8 and len({r["rollout_id"] for r in records})==8'
    source = source.replace(needle, needle + '\n            diagnostics = [r["harness_exit_diagnostics"] for r in records if "harness_exit_diagnostics" in r]\n            assert diagnostics == ' + repr([EXPECTED]) + '\n            assert "dummy-private-token" not in json.dumps(records)\n            assert "private.invalid" not in json.dumps(records)')
    namespace = {"__name__": "retention_http_fixture"}
    exec(compile(source, "retention_http_fixture", "exec"), namespace)
    await namespace["main"]()
    proof = {"passed": True, "actual_harness_response_boundaries": tested, "actual_http_audit_records": 8, "mask_and_identity_unchanged": True, "raw_log_secrets_excluded": True, "gpu_or_live_sandbox_test": False}
    print(json.dumps(proof))


if __name__ == "__main__":
    asyncio.run(main())
