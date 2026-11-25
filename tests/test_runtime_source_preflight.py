import hashlib
import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('runtime_sources', Path(__file__).parents[1] / 'scripts/check_runtime_sources.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def test_ready_includes_intentionally_new_files(tmp_path):
    (tmp_path / 'existing').write_bytes(b'before')
    manifest = {'files': {'existing': {'before_sha256': digest(b'before'), 'after_sha256': digest(b'after')},
                          'new': {'before_sha256': None, 'after_sha256': digest(b'new')}}}
    assert module.check_profile(tmp_path, manifest)['status'] == 'ready'
    (tmp_path / 'existing').write_bytes(b'after')
    assert module.check_profile(tmp_path, manifest)['status'] == 'incompatible'
    (tmp_path / 'new').write_bytes(b'new')
    assert module.check_profile(tmp_path, manifest)['status'] == 'applied'


def test_unrelated_source_is_not_accepted(tmp_path):
    (tmp_path / 'file').write_bytes(b'unrelated')
    manifest = {'files': {'file': {'before_sha256': digest(b'old'), 'after_sha256': digest(b'new')}}}
    assert module.check_profile(tmp_path, manifest)['files'][0]['state'] == 'different'
    (tmp_path / 'file').unlink()
    assert module.check_profile(tmp_path, manifest)['files'][0]['state'] == 'missing'


def test_manifest_cannot_escape_root(tmp_path):
    with pytest.raises(ValueError):
        module.check_profile(tmp_path, {'files': {'../escape': {}}})
