"""Verify mounted kernels and load a pinned short-sequence reference for tests."""
import ast
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys


def verify_and_reference(baseline_root):
    repo = Path(__file__).resolve().parents[1]
    manifest = json.loads((repo / 'patches/mamba-long-sequence-indexing.json').read_text())
    for name, version in manifest['runtime_versions'].items():
        assert importlib.metadata.version(name) == version, name
    references = []
    observed = {}
    for filename, entry in manifest['files'].items():
        installed = importlib.import_module('mamba_ssm.ops.triton.' + Path(filename).stem)
        observed[filename] = hashlib.sha256(Path(installed.__file__).read_bytes()).hexdigest()
        assert observed[filename] == entry['after_sha256'], 'Missing installed patch: ' + filename
        source = baseline_root / filename
        assert hashlib.sha256(source.read_bytes()).hexdigest() == entry['before_sha256'], filename
        name = 'mamba_ssm.ops.triton.mmrl_test_reference_' + Path(filename).stem
        spec = importlib.util.spec_from_file_location(name, source)
        reference = importlib.util.module_from_spec(spec)
        sys.modules[name] = reference
        spec.loader.exec_module(reference)
        for function in ast.parse(source.read_text()).body:
            if not isinstance(function, ast.FunctionDef):
                continue
            uses_program_id = any(
                isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'program_id' for node in ast.walk(function)
            )
            if uses_program_id or function.name == 'ensure_stride':
                references.append((installed, function.name, getattr(reference, function.name)))

    def use_baseline():
        for module, name, function in references:
            setattr(module, name, function)

    return use_baseline, observed
