"""Optional project adapter for the external Gym harness agent."""

from nemo_gym.openai_utils import NeMoGymResponse
from responses_api_agents.harness_agent.app import HarnessAgent, HarnessAgentConfig

from mmrl.harness_failure import runner_failure_response


class Config(HarnessAgentConfig):
    runner_failure_as_response: bool = True


class CodingHarnessAgent(HarnessAgent):
    config: Config

    async def responses(self, request, body):
        try:
            return await super().responses(request, body)
        except RuntimeError as error:
            if not self.config.runner_failure_as_response:
                raise
            response = runner_failure_response(error, body, NeMoGymResponse)
            if response is None:
                raise
            return response
