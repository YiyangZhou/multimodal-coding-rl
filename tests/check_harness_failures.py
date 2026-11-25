"""Check native unscoped OpenCode exit status at the remote runner boundary."""

import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from responses_api_agents.opencode_agent.app import OpenCodeAgent, OpenCodeAgentConfig


class OpenCodeExitTests(unittest.IsolatedAsyncioTestCase):
    async def check_exit(self, returncode, expected, *, timeout=False):
        with tempfile.TemporaryDirectory() as directory:
            config = OpenCodeAgentConfig(
                host="127.0.0.1",
                port=8001,
                entrypoint="app.py",
                name="probe",
                resources_server={"type": "resources_servers", "name": "unused"},
                workspace_root=directory,
                command="opencode",
                timeout=1,
            )
            with patch("responses_api_agents.opencode_agent.app.ensure_opencode"):
                agent = OpenCodeAgent(
                    config=config, server_client=MagicMock(spec=ServerClient)
                )
            process = MagicMock(returncode=returncode)
            process.communicate = AsyncMock(
                return_value=(b"", b"synthetic process result")
            )
            if timeout:
                process.communicate.side_effect = [TimeoutError(), (b"", b"timeout")]
            with patch(
                "asyncio.create_subprocess_exec", AsyncMock(return_value=process)
            ):
                response = await agent.responses(
                    SimpleNamespace(path_params={}),
                    NeMoGymResponseCreateParamsNonStreaming(input="Complete the task"),
                )
            self.assertEqual(response.status, expected)
            self.assertEqual(response.model_dump(mode="json")["status"], expected)
            if timeout:
                process.kill.assert_called_once()

    async def test_unscoped_nonzero_exit_is_failed(self):
        await self.check_exit(1, "failed")

    async def test_unscoped_success_is_completed(self):
        await self.check_exit(0, "completed")

    async def test_unscoped_timeout_is_incomplete(self):
        await self.check_exit(-9, "incomplete", timeout=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
