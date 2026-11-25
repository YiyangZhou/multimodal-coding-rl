"""Reject zero-exit OpenCode sessions that stop at an unresolved tool turn."""
import json
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from mmrl.config import gym_config
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from responses_api_agents.opencode_agent.app import OpenCodeAgent, OpenCodeAgentConfig


class OpenCodeTerminalTests(unittest.IsolatedAsyncioTestCase):
    async def run_session(self, finishes, expected, *, child_finish=None, api_error=False):
        with tempfile.TemporaryDirectory() as directory:
            cfg = OpenCodeAgentConfig(
                host="127.0.0.1", port=8001, name="terminal-test", entrypoint="app.py",
                resources_server={"type": "resources_servers", "name": "unused"},
                workspace_root=directory, command="opencode", timeout=1,
            )
            with patch("responses_api_agents.opencode_agent.app.ensure_opencode"):
                agent = OpenCodeAgent(config=cfg, server_client=MagicMock(spec=ServerClient))

            async def spawn(*args, env, **kwargs):
                db = Path(env["XDG_DATA_HOME"]) / "opencode/opencode.db"
                db.parent.mkdir(parents=True, exist_ok=True)
                with sqlite3.connect(db) as con:
                    con.executescript("CREATE TABLE session(id TEXT, parent_id TEXT);"
                                      "CREATE TABLE message(session_id TEXT, time_created INT, data TEXT);")
                    con.execute("INSERT INTO session VALUES ('root', NULL)")
                    for i, finish in enumerate(finishes):
                        data = {"role": "assistant", "finish": finish}
                        if api_error:
                            data["error"] = {"name": "APIError"}
                        con.execute("INSERT INTO message VALUES ('root', ?, ?)", (i, json.dumps(data)))
                    if child_finish:
                        con.execute("INSERT INTO session VALUES ('child', 'root')")
                        con.execute("INSERT INTO message VALUES ('child', 99, ?)",
                                    (json.dumps({"role": "assistant", "finish": child_finish}),))
                return MagicMock(returncode=0, communicate=AsyncMock(return_value=(b"", b"")))

            # Exercise the actual root-session SQL guard and unscoped response
            # status used by the remote runner. Content conversion is orthogonal.
            with patch("asyncio.create_subprocess_exec", side_effect=spawn), patch(
                "responses_api_agents.opencode_agent.app.parse_opencode_session",
                return_value=([], {"input_tokens": 0, "output_tokens": 0}),
            ):
                response = await agent.responses(SimpleNamespace(path_params={}),
                                                NeMoGymResponseCreateParamsNonStreaming(input="Inspect the renderer"))
            self.assertEqual(response.status, expected)
            self.assertEqual(response.model_dump(mode="json")["status"], expected)

    async def test_permission_abort_with_zero_exit_is_failed(self):
        await self.run_session(["tool-calls"], "failed")

    async def test_multiple_tool_turns_then_terminal_answer_complete(self):
        await self.run_session(["tool-calls", "tool-calls", "stop"], "completed")

    async def test_terminal_answer_before_later_aborted_turn_does_not_hide_abort(self):
        await self.run_session(["stop", "tool-calls"], "failed")

    async def test_child_session_does_not_replace_root_terminal_state(self):
        await self.run_session(["stop"], "completed", child_finish="tool-calls")

    async def test_api_error_is_still_failed(self):
        await self.run_session(["stop"], "failed", api_error=True)

    async def test_only_renderer_and_sandbox_scratch_directories_are_allowed(self):
        cfg = gym_config("registry.example/image@sha256:" + "a" * 64, "https://model.example")
        permissions = cfg["mm_opencode"]["responses_api_agents"]["harness_agent"]["agent_kwargs"]["opencode_config"]["permission"]
        self.assertEqual(permissions["external_directory"], {"/opt/mmrl/*": "allow", "/tmp/*": "allow"})
        self.assertEqual(permissions["webfetch"], "deny")


if __name__ == "__main__":
    unittest.main(verbosity=2)
