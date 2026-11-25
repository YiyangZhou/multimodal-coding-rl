import pytest

@pytest.fixture(autouse=True)
def synthetic_judge_model(monkeypatch):
    # Unit tests never use a production scoring model.
    monkeypatch.setenv("MMRL_JUDGE_MODEL", "unit-test-judge")
