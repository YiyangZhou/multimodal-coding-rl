import json
from collections import Counter

import pytest
from datasets import Dataset

from mmrl.balancing import ELEVEN_BENCHMARKS, balanced_indices
from mmrl.config import training_config


def rows():
    return [json.dumps({
        "task": {"benchmark": name, "task_id": str(i)},
        "route": "single", "agent_ref": {"name": "mm_single"},
        "responses_create_params": {"max_output_tokens": 32768},
    }) for count, name in enumerate(ELEVEN_BENCHMARKS, 1) for i in range(count)]


def test_unequal_sources_get_equal_exposure_without_losing_source_rows():
    source = rows()
    indices, report = balanced_indices(source, seed=17, batch_size=8)
    assert set(indices) == set(range(len(source)))
    counts = Counter(json.loads(source[i])["task"]["benchmark"] for i in indices)
    assert counts == dict.fromkeys(ELEVEN_BENCHMARKS, 16)
    assert len(indices) % 8 == 0
    assert report["source_counts"][ELEVEN_BENCHMARKS[-1]] == 11
    assert report["weight_each"] == pytest.approx(1 / 11)
    assert indices == balanced_indices(source, seed=17, batch_size=8)[0]
    assert indices != balanced_indices(source, seed=18, batch_size=8)[0]
    # Real Arrow selection preserves every source payload, including repeated
    # entries, without materializing duplicated underlying data buffers.
    dataset = Dataset.from_dict({"extra_env_info": source})
    selected = dataset.select(indices)
    assert selected["extra_env_info"][:] == [source[i] for i in indices]
    assert selected._data.nbytes == dataset._data.nbytes


@pytest.mark.parametrize("change,match", [
    (lambda r: r[1:], "Missing benchmarks"),
    (lambda r: r + [r[0]], "Duplicate task"),
    (lambda r: [], "Missing benchmarks"),
    (lambda r: [r[0].replace('"single"', '"codex"')] + r[1:], "single-turn"),
    (lambda r: [r[0].replace('"mm_single"', '"mm_codex"')] + r[1:], "single-turn"),
    (lambda r: [r[0].replace('"chart2code"', '"unknown"')] + r[1:], "Unknown benchmark"),
    (lambda r: [r[0].replace('"task_id": "0"', '"task_id": ""')] + r[1:], "identity"),
])
def test_incomplete_or_wrong_data_cannot_silently_change_sampling(change, match):
    with pytest.raises(ValueError, match=match):
        balanced_indices(change(rows()))


@pytest.mark.parametrize("kwargs", [{"seed": True}, {"batch_size": False}, {"batch_size": 0}])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        balanced_indices(rows(), **kwargs)


def base_config():
    return {
        "data_plane": {"enabled": True}, "async_rl": {"max_inflight_rollouts": 56},
        "policy": {"generation": {"temperature": 1., "top_p": 1., "top_k": None,
                                   "max_new_tokens": 32768, "vllm_cfg": {"max_model_len": 65536}},
                   "train_global_batch_size": 128, "optimizer": {"lr": 5e-7}},
        "data": {"max_input_seq_length": 32768},
        "grpo": {"num_prompts_per_step": 8, "num_generations_per_prompt": 16},
        "checkpointing": {}, "logger": {},
    }


def test_configuration_preserves_normal_training_parameters(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text("\n".join(rows()) + "\n")
    base = base_config()
    cfg = training_config(base, "/gym.json", path, "/run", balanced_eleven=True)
    assert cfg["policy"] == base["policy"]
    assert cfg["async_rl"]["max_inflight_rollouts"] == 56
    assert cfg["grpo"]["num_prompts_per_step"] == 8
    assert cfg["grpo"]["num_generations_per_prompt"] == 16
    assert cfg["data"]["train"]["balance_batch_size"] == 8
    assert cfg["data"]["train"]["dataset_name"] == "mmrl.balanced_dataset.ElevenBenchmarkGymDataset"
    assert cfg["data"]["shuffle"] and not cfg["data"]["use_multiple_dataloader"]
    assert "train" not in base["data"]


def test_cli_rejects_missing_benchmarks_before_creating_output(tmp_path, monkeypatch):
    import sys
    from mmrl.cli import main

    data, base, output = tmp_path / "data.jsonl", tmp_path / "base.json", tmp_path / "output"
    data.write_text(rows()[0] + "\n")
    base.write_text(json.dumps(base_config()))
    monkeypatch.setattr(sys, "argv", [
        "mmrl", "compose", "--base", str(base), "--data", str(data),
        "--image", "registry.test/image@sha256:" + "a" * 64,
        "--gateway", "auto", "--output", str(output), "--balanced-eleven",
    ])
    with pytest.raises(ValueError, match="Missing benchmarks"):
        main()
    assert not output.exists()
