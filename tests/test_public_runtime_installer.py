"""Exercise installation guards with real Git revisions and patch execution."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.fixture
def install_fixture(tmp_path):
    package = tmp_path / 'package'
    (package / 'scripts').mkdir(parents=True)
    (package / 'patches').mkdir()
    script = package / 'scripts/apply_capture_patch.py'
    shutil.copyfile(Path(__file__).parents[1] / 'scripts/apply_capture_patch.py', script)
    root = tmp_path / 'runtime'
    root.mkdir()
    subprocess.run(['git', 'init', '-q', str(root)], check=True)
    (root / 'source.py').write_text('old = 1\n')
    subprocess.run(['git', '-C', str(root), 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(root), '-c', 'user.name=Test', '-c', 'user.email=test@example.org', 'commit', '-qm', 'fixture'], check=True)
    commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    patch = b'--- a/source.py\n+++ b/source.py\n@@ -1 +1 @@\n-old = 1\n+new = 2\n'
    digest = lambda b: hashlib.sha256(b).hexdigest()
    manifest = {'source_commits': {'.': commit}, 'files': {'source.py': {'before_sha256': digest(b'old = 1\n'), 'after_sha256': digest(b'new = 2\n')}}, 'patch_sha256': digest(patch)}
    (package / 'patches/public-native-runtime.patch').write_bytes(patch)
    manifest_path = package / 'patches/public-native-runtime.json'
    manifest_path.write_text(json.dumps(manifest))
    def run(*extra):
        return subprocess.run([sys.executable, str(script), '--rl-root', str(root), '--public-runtime', *extra], capture_output=True, text=True)
    return root, manifest, manifest_path, run


def test_apply_reapply_and_check(install_fixture):
    root, _, _, run = install_fixture
    assert run('--check').returncode != 0
    assert run().returncode == 0
    assert (root / 'source.py').read_text() == 'new = 2\n'
    assert run().returncode == 0
    assert run('--check').returncode == 0


def test_wrong_revision_rejected_before_write(install_fixture):
    root, manifest, path, run = install_fixture
    manifest['source_commits']['.'] = '0' * 40
    path.write_text(json.dumps(manifest))
    result = run()
    assert result.returncode != 0 and 'Source revision mismatch' in result.stderr
    assert (root / 'source.py').read_text() == 'old = 1\n'


def test_modified_source_rejected_before_write(install_fixture):
    root, _, _, run = install_fixture
    (root / 'source.py').write_text('user_change = 3\n')
    result = run()
    assert result.returncode != 0 and 'partially patched snapshot' in result.stderr
    assert (root / 'source.py').read_text() == 'user_change = 3\n'


def test_modified_patch_rejected_before_write(install_fixture):
    root, _, path, run = install_fixture
    path.with_suffix('.patch').write_text('invalid patch\n')
    result = run()
    assert result.returncode != 0 and 'does not match its manifest' in result.stderr
    assert (root / 'source.py').read_text() == 'old = 1\n'
