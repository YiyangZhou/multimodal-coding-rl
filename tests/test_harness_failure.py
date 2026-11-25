from types import SimpleNamespace

import pytest

from mmrl.harness_failure import runner_failure_response


def params():
    return SimpleNamespace(model='test-policy', tool_choice='auto', tools=[], parallel_tool_calls=False)


def test_runner_failure_is_failed_and_does_not_expose_runner_logs():
    result = runner_failure_response(RuntimeError('runner failed (1): private log payload'), params(), SimpleNamespace)
    assert result.status == 'failed'
    assert result.output == []
    assert result.error == {'code': 'server_error', 'message': 'harness_runner_failed'}
    assert result.metadata == {'failed': 'true'}
    assert 'private log payload' not in repr(result)


@pytest.mark.parametrize('error', [RuntimeError('model setup failed'), ValueError('runner failed (1)'), ConnectionError('connection lost'), TimeoutError('request timed out')])
def test_unrelated_errors_are_not_converted(error):
    assert runner_failure_response(error, params(), SimpleNamespace) is None


def test_project_adapter_changes_component_only():
    from mmrl.config import gym_config
    args = ('registry.example/image@sha256:' + 'a' * 64, 'https://capture.example')
    original = gym_config(*args)
    adapted = gym_config(*args, project_harness_adapter=True)
    for harness in ('opencode', 'codex', 'hermes'):
        original_config = original[f'mm_{harness}']['responses_api_agents']['harness_agent']
        adapted_config = adapted[f'mm_{harness}']['responses_api_agents']['mm_harness']
        assert adapted_config == original_config
    assert adapted['mm_coding'] == original['mm_coding']
