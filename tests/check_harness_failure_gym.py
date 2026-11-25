"""CPU contract check against an installed external Gym; no sandbox rollout."""
import asyncio
from unittest.mock import AsyncMock, patch
from types import SimpleNamespace
from dataclasses import asdict

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from responses_api_agents.harness_agent.app import HarnessAgent
from mmrl.harness_app import CodingHarnessAgent, Config


async def check():
    body = NeMoGymResponseCreateParamsNonStreaming(input='test', model='policy')
    # Avoid constructing sandbox providers; test only the real response-model boundary.
    with patch.object(CodingHarnessAgent, "model_post_init", lambda self, context: None):
        agent = CodingHarnessAgent.model_construct(config=Config.model_construct(runner_failure_as_response=True))
    with patch.object(HarnessAgent, 'responses', new=AsyncMock(side_effect=RuntimeError('runner failed (1): private log'))):
        response = await agent.responses(None, body)
    assert response.status == 'failed' and response.output == []
    assert response.error.message == 'harness_runner_failed'
    assert 'private log' not in response.model_dump_json()
    from mmrl.gym_app import CodingResourcesServer, VerifyRequest
    from mmrl.tasks import Task
    task = Task(benchmark='mmsvg_text2svg', task_id='failure-contract', prompt='Draw a circle')
    verify_body = VerifyRequest(responses_create_params=body, response=response,
                                task=asdict(task), route='opencode')
    session = SimpleNamespace(task=task, route='opencode', handle=None,
                              capture_rollout_id=verify_body.capture_rollout_id)
    service = SimpleNamespace(
        _pop_session=lambda key: session,
        _before_verify=lambda current: None,
        config=SimpleNamespace(verification_log_dir=None),
    )
    result = await CodingResourcesServer.verify(service, SimpleNamespace(session={'mm_session': 'test'}), verify_body)
    assert result.reward == 0.0 and result.mask_sample is True
    assert result.instance_config['mask_sample'] is True
    assert result.failure_kind == 'mm_coding:infrastructure'

    for error in (RuntimeError('unrelated'), ConnectionError('offline')):
        with patch.object(HarnessAgent, 'responses', new=AsyncMock(side_effect=error)):
            try:
                await agent.responses(None, body)
            except Exception as actual:
                assert actual is error
            else:
                raise AssertionError('Unexpectedly suppressed unrelated error')
    agent.config.runner_failure_as_response = False
    error = RuntimeError('runner failed (1): private log')
    with patch.object(HarnessAgent, 'responses', new=AsyncMock(side_effect=error)):
        try:
            await agent.responses(None, body)
        except RuntimeError as actual:
            assert actual is error
        else:
            raise AssertionError('Disabled adapter suppressed runner failure')
    print('4 external Gym adapter contracts and resource failure masking passed; no real sandbox or model request')


if __name__ == '__main__':
    asyncio.run(check())
