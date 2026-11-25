"""Exercise the source-guarded Mamba installer against an unpatched snapshot."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

p = argparse.ArgumentParser()
p.add_argument('--baseline', type=Path, required=True)
args = p.parse_args()
repo = Path(__file__).resolve().parents[1]
manifest = json.loads((repo / 'patches/mamba-long-sequence-indexing.json').read_text())

def hashes(directory):
    return {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in manifest['files']}

def run(directory, check=False):
    command = [sys.executable, str(repo / 'scripts/apply_capture_patch.py'), '--mamba-root', str(directory)]
    return subprocess.run(command + (['--check'] if check else []), capture_output=True, text=True)

with tempfile.TemporaryDirectory(prefix='mmrl-mamba-patch-') as temporary:
    root = Path(temporary)
    good, wrong = root / 'good', root / 'wrong'
    for directory in [good, wrong]:
        directory.mkdir()
        for name, spec in manifest['files'].items():
            source = args.baseline / name
            assert hashlib.sha256(source.read_bytes()).hexdigest() == spec['before_sha256'], name
            shutil.copyfile(source, directory / name)
    assert run(good, check=True).returncode != 0
    result = run(good)
    assert result.returncode == 0, result.stderr + result.stdout
    expected = {name: spec['after_sha256'] for name, spec in manifest['files'].items()}
    assert hashes(good) == expected
    assert run(good, check=True).returncode == 0
    assert run(good).returncode == 0
    assert hashes(good) == expected
    with (wrong / 'layernorm_gated.py').open('a') as stream:
        stream.write('\n# Different source snapshot\n')
    before = hashes(wrong)
    result = run(wrong)
    assert result.returncode != 0 and 'Refusing a different' in result.stderr
    assert hashes(wrong) == before
print(json.dumps({'passed': True, 'checks': ['reject_unapplied_check', 'exact_patch_replay',
    'check_applied', 'idempotent_reapply', 'reject_wrong_source_without_mutation'],
    'gpu_acceptance': False}))
