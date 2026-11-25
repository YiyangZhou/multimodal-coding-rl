import pytest
from mmrl.reward import QualityJudge, InfrastructureError


def test_explicit_provider_configuration(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEYS", "test-key")
    monkeypatch.setenv("MMRL_JUDGE_MODEL", "example-vision-model")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://judge.example/v1/")
    judge = QualityJudge()
    assert judge.model == "example-vision-model"
    assert judge.responses_url == "https://judge.example/v1/responses"


def test_missing_model_is_configuration_error(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEYS", "test-key")
    monkeypatch.delenv("MMRL_JUDGE_MODEL")
    with pytest.raises(InfrastructureError, match="MMRL_JUDGE_MODEL"):
        QualityJudge()


def test_single_key_and_public_endpoint(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEYS", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    judge = QualityJudge()
    assert judge._keys == ("test-key",)
    assert judge.responses_url == "https://api.openai.com/v1/responses"
