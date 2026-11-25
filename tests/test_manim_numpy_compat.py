"""Renderer compatibility must preserve the numerical integration contract."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


def entry(monkeypatch):
    folder = Path(__file__).parents[1] / 'mmrl'
    monkeypatch.syspath_prepend(str(folder))
    spec = importlib.util.spec_from_file_location('manim_cli_entry', folder / 'manim_entry.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_missing_trapz_preserves_nonuniform_and_signed_integrals(monkeypatch):
    module = entry(monkeypatch)
    monkeypatch.delattr(np, 'trapz', raising=False)
    # Make restoration remove the alias even on NumPy versions without trapz.
    monkeypatch.setattr(np, 'trapz', None, raising=False)
    monkeypatch.delattr(np, 'trapz')
    module.install_numpy_compat()
    assert np.trapz([0, 1, 3], x=[0, 1, 3]) == pytest.approx(4.5)
    assert np.trapz([3, 1, 0], x=[3, 1, 0]) == pytest.approx(-4.5)
    np.testing.assert_allclose(np.trapz([[0, 1, 3], [0, 2, 6]], x=[0, 1, 3], axis=1), [4.5, 9.0])
    module.install_numpy_compat()
    assert np.trapz is np.trapezoid


def test_existing_trapz_is_not_replaced(monkeypatch):
    module = entry(monkeypatch)
    sentinel = object()
    monkeypatch.setattr(np, 'trapz', sentinel, raising=False)
    module.install_numpy_compat()
    assert np.trapz is sentinel
