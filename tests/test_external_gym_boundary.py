"""The public tree must never silently fall back to bundled Gym patches."""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_no_bundled_gym_runtime():
    assert not (ROOT / 'gym').exists()
    assert not list((ROOT / 'patches').glob('gym-*'))
    adapter = ROOT / 'integrations/nemo_gym/resources_servers/mm_coding/app.py'
    assert 'from mmrl.gym_app import CodingResourcesServer' in adapter.read_text()


def test_gym_patch_check_requires_external_manifest(tmp_path):
    env = dict(os.environ)
    env.pop('MMRL_EXTERNAL_GYM_PATCH_DIR', None)
    result = subprocess.run(
        [sys.executable, str(ROOT / 'scripts/apply_capture_patch.py'),
         '--gym-root', str(tmp_path), '--check'], env=env,
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert 'Gym patches are not distributed here' in result.stderr
    assert not list(tmp_path.iterdir())


def test_history_check_requires_external_manifest():
    env = dict(os.environ)
    env.pop('MMRL_EXTERNAL_GYM_PATCH_DIR', None)
    result = subprocess.run(
        [sys.executable, str(ROOT / 'scripts/check_history_imports.py')],
        env=env, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert 'MMRL_EXTERNAL_GYM_PATCH_DIR' in result.stderr
    assert 'ModuleNotFoundError' not in result.stderr


def test_image_builder_requires_external_source_lock(tmp_path):
    result = subprocess.run(
        [sys.executable, str(ROOT / 'scripts/prepare_image.py'),
         '--gym-root', str(tmp_path), '--hermes-root', str(tmp_path),
         '--output', str(tmp_path / 'output')],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert '--gym-source-lock' in result.stderr
    assert not (tmp_path / 'output').exists()


def test_source_lock_rejects_paths_outside_checkout(tmp_path):
    import importlib.util
    import json
    import pytest
    spec = importlib.util.spec_from_file_location('prepare_image', ROOT / 'scripts/prepare_image.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source_lock = tmp_path / 'lock.json'
    for pins in ({}, {'../private': 'a' * 64}, {'/private': 'a' * 64}, {'pyproject.toml': 'bad'}):
        source_lock.write_text(json.dumps(pins))
        with pytest.raises(ValueError):
            module.load_source_pins(source_lock)
    pins = {'pyproject.toml': 'a' * 64, 'uv.lock': 'b' * 64}
    source_lock.write_text(json.dumps(pins))
    assert module.load_source_pins(source_lock) == pins
