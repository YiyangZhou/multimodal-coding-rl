import hashlib
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from download_benchmarks import digest, target, verified
from prepare_manimbench import prepare


def test_download_integrity_rejects_same_size_corruption(tmp_path):
    file = tmp_path / "data"
    file.write_bytes(b"good")
    pin = {"size": 4, "hash_algorithm": "sha256", "hash": hashlib.sha256(b"good").hexdigest()}
    assert verified(file, pin)
    file.write_bytes(b"evil")
    assert not verified(file, pin)


def test_git_blob_hash_for_non_lfs_files(tmp_path):
    file = tmp_path / "small"
    file.write_bytes(b"hello\n")
    assert digest(file, "sha1") == hashlib.sha1(b"blob 6\0hello\n").hexdigest()


def test_paths_cannot_escape_dataset_root(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    (root / "escape").symlink_to(tmp_path, target_is_directory=True)
    for path in ("../outside", "/outside", "escape/outside"):
        with pytest.raises(ValueError):
            target(root, path)


def test_test_set_preserved_and_overlapping_train_rows_excluded():
    def row(prompt, code, split):
        return {"Reviewed Description": prompt, "Code": code, "Split": split, "Type": "Basic"}
    train = [row("same", "different implementation", "train"),
             row("different", "same code", "train"), row("independent", "private answer", "train")]
    test = [row("same", "same code", "test")]
    prepared, excluded = prepare(train, test)
    assert len(prepared["train"]) == len(prepared["test"]) == 1
    assert [x["reasons"] for x in excluded] == [["exact_test_prompt"], ["exact_test_code"]]
    assert prepared["train"][0]["prompt"] == "independent"
    assert prepared["train"][0]["ground_truth"] == "private answer"
    assert prepared["test"][0]["prompt"] == test[0]["Reviewed Description"]


def test_declared_split_mismatch_is_not_silently_relabelled():
    with pytest.raises(ValueError, match="split mismatch"):
        prepare([{"Reviewed Description": "task", "Code": "answer", "Split": "test"}], [])
