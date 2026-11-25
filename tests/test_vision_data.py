import json
from pathlib import Path
import runpy

from PIL import Image
import pytest

from mmrl.vision_data import prepare_project, public_input, verify_manifest


def project(tmp_path, level="webpage"):
    root = tmp_path / level / "counter"
    (root / "prototypes").mkdir(parents=True)
    (root / "resources" / "icons").mkdir(parents=True)
    Image.new("RGB", (16, 16), "red").save(root / "prototypes/home.jpg")
    (root / "resources/icons/logo.svg").write_text("<svg>public asset</svg>")
    (root / "workflow.json").write_text(json.dumps([
        {"content": [{"objective": "PRIVATE_CHECK", "actions": [], "validations": ["PRIVATE_ASSERTION"]}],
         "prototype": {"home": {"idx": 1}}},
        {"prototype": {"home": {"idx": 0}}},
    ]))
    if level == "frontend":
        (root / "prompt.txt").write_text("Implement the specified interactive counter.")
    if level == "website":
        (root / "prd.md").write_text("Implement database persistence and a counter API.")
    return root


@pytest.mark.parametrize("level", ["webpage", "frontend", "website"])
def test_all_levels_keep_assets_and_private_workflows_separate(tmp_path, level):
    manifest = prepare_project(project(tmp_path, level), level)
    assert manifest["upstream_split"] == "test"
    assert [x["workflow"] for x in manifest["private_visual_references"]] == [0, 1]
    public = json.dumps(public_input(manifest))
    assert "PRIVATE_CHECK" not in public and "PRIVATE_ASSERTION" not in public
    assert "workflow.json" not in public
    assert "resources/icons/logo.svg" in public and "prototypes/home.jpg" in public
    if level == "website":
        assert "database persistence" in public


@pytest.mark.parametrize("changed", ["resources/icons/logo.svg", "prototypes/home.jpg", "workflow.json", "prd.md"])
def test_content_drift_invalidates_manifest(tmp_path, changed):
    root = project(tmp_path, "website")
    manifest = prepare_project(root, "website")
    with (root / changed).open("ab") as stream:
        stream.write(b"\n ")
    with pytest.raises(ValueError):
        verify_manifest(manifest)


def test_new_unrecorded_resource_invalidates_manifest(tmp_path):
    root = project(tmp_path)
    manifest = prepare_project(root, "webpage")
    (root / "resources/extra.txt").write_text("new")
    with pytest.raises(ValueError, match="changed"):
        verify_manifest(manifest)


def test_missing_required_full_stack_prd_is_not_silently_dropped(tmp_path):
    root = project(tmp_path, "website")
    (root / "prd.md").unlink()
    with pytest.raises(ValueError):
        prepare_project(root, "website")


def test_missing_named_checkpoint_cannot_use_another_image(tmp_path):
    root = project(tmp_path)
    (root / "prototypes/home.jpg").rename(root / "prototypes/other.jpg")
    with pytest.raises(ValueError, match="exact upstream"):
        prepare_project(root, "webpage")


@pytest.mark.parametrize("name", ["resources/linked", "resources/icons/logo.svg", "workflow.json"])
def test_no_symlink_can_smuggle_host_files_into_public_assets(tmp_path, name):
    root = project(tmp_path)
    target = tmp_path / "private.txt"
    target.write_text("PRIVATE")
    path = root / name
    path.unlink(missing_ok=True)
    path.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        prepare_project(root, "webpage")


def test_manifest_tampering_cannot_change_policy_or_judge_inputs(tmp_path):
    manifest = prepare_project(project(tmp_path), "webpage")
    manifest["private_workflows"][0]["content"][0]["validations"] = ["Weakened condition"]
    with pytest.raises(ValueError, match="changed"):
        public_input(manifest)


def test_preparation_keeps_all_three_levels_and_never_invents_train_split(tmp_path):
    for level in ("webpage", "frontend", "website"):
        project(tmp_path / "data", level)
    prepare = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/prepare_vision2web.py"))["prepare"]
    report = prepare(tmp_path / "data", tmp_path / "prepared")
    assert report["counts"] == {"webpage": 1, "frontend": 1, "website": 1}
    assert report["upstream_split"] == "test" and report["training_ready"] is False
    assert len(report["files_sha256"]) == 3
    assert not list((tmp_path / "prepared").glob("*train*"))
    with pytest.raises(FileExistsError):
        prepare(tmp_path / "data", tmp_path / "prepared")


def test_preparation_never_silently_discards_a_broken_level(tmp_path):
    project(tmp_path / "data", "webpage")
    prepare = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/prepare_vision2web.py"))["prepare"]
    with pytest.raises(ValueError, match="Missing official level"):
        prepare(tmp_path / "data", tmp_path / "prepared")
    assert not (tmp_path / "prepared").exists()


def test_quarantine_missing_prototype_is_explicit_and_recorded(tmp_path):
    import shutil
    for level in ("webpage", "frontend", "website"):
        project(tmp_path / "data", level)
    bad = tmp_path / "data/frontend/broken"
    shutil.copytree(tmp_path / "data/frontend/counter", bad)
    (bad / "prototypes/home.jpg").rename(bad / "prototypes/other.jpg")
    prepare = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/prepare_vision2web.py"))["prepare"]
    with pytest.raises(ValueError, match="exact upstream"):
        prepare(tmp_path / "data", tmp_path / "strict")
    assert not (tmp_path / "strict").exists()
    report = prepare(tmp_path / "data", tmp_path / "prepared", quarantine_missing_prototypes=True)
    assert report["discovered_projects"] == 4
    assert sum(report["counts"].values()) == 3
    assert report["quarantined"][0]["task_id"] == "vision2web/frontend/broken"
    assert report["quarantined"][0]["missing_asset"] == "prototypes/home.jpg"
    assert len(report["quarantined"][0]["workflow_sha256"]) == 64
    assert not (tmp_path / "prepared/frontend/broken.json").exists()
    # Other data errors must still abort; the option is not a general skip switch.
    (bad / "workflow.json").write_text("bad JSON")
    with pytest.raises(ValueError):
        prepare(tmp_path / "data", tmp_path / "bad-json", quarantine_missing_prototypes=True)
    assert not (tmp_path / "bad-json").exists()
