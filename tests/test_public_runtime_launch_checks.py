"""Verify composite selection preserves native and external Gym launch gates."""
import json
from pathlib import Path
import runpy
import sys

import pytest


@pytest.mark.parametrize('history', [False, True])
def test_public_profile_checks_composite_and_keeps_gym_gate(tmp_path, monkeypatch, history):
    namespace = runpy.run_path(str(Path(__file__).parents[1] / 'scripts/train.py'))
    main = namespace['main']
    globals_ = main.__globals__
    calls = []
    class StopBeforeServices(Exception):
        pass
    def stop():
        raise StopBeforeServices
    for name in ('validate_thinking_parser', 'validate_capture_retries', 'validate_sibling_recipe'):
        monkeypatch.setitem(globals_, name, lambda cfg: None)
    monkeypatch.setitem(globals_, 'original_history_root', lambda cfg, env: '/capture' if history else None)
    monkeypatch.setitem(globals_, 'provider_config', stop)
    monkeypatch.setattr(globals_['subprocess'], 'run', lambda command, **kw: calls.append(command))
    monkeypatch.setenv('NEMO_GYM_ROOT', str(tmp_path / 'gym'))
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'data': {'train': {'dataset_name': 'mmrl.mixed_dataset.BalancedMixedGymDataset'}}}))
    monkeypatch.setattr(sys, 'argv', ['train.py', '--config', str(config), '--rl-root', str(tmp_path), '--public-runtime', '--check-only'])
    with pytest.raises(StopBeforeServices):
        main()
    assert len(calls) == (2 if history else 1)
    assert '--public-runtime' in calls[0] and '--check' in calls[0]
    if history:
        assert '--gym-root' in calls[1] and '--original-history' in calls[1]
        assert '--check' in calls[1]
