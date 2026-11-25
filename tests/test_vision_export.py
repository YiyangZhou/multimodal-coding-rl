import json
from pathlib import Path
import runpy

from PIL import Image
import pytest

from mmrl.vision_data import LEVELS

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
prepare = runpy.run_path(str(SCRIPTS / "prepare_vision2web.py"))["prepare"]
export = runpy.run_path(str(SCRIPTS / "export_vision_gym.py"))["export"]


def inventory(tmp_path):
    assignment = {"train": [], "validation": []}
    for index, level in enumerate(LEVELS):
        for number, split in enumerate(assignment):
            name = f"project-{index}-{number}"
            root = tmp_path / "data" / level / name
            (root / "prototypes").mkdir(parents=True)
            Image.new("RGB", (16, 16), (index * 70, number * 150, 120)).save(root / "prototypes/home.jpg")
            (root / "workflow.json").write_text(json.dumps([{
                "content": [{"objective": "PRIVATE_OBJECTIVE", "actions": [],
                             "validations": ["PRIVATE_ASSERTION"]}],
                "prototype": {"home": {"idx": 1}},
            }]))
            if level != "webpage":
                (root / ("prompt.txt" if level == "frontend" else "prd.md")).write_text(f"Build {name}.")
            assignment[split].append(f"vision2web/{level}/{name}")
    prepared = tmp_path / "prepared"
    prepare(tmp_path / "data", prepared)
    return prepared, assignment


def test_complete_custom_export_is_single_turn_with_private_tests_excluded(tmp_path):
    prepared, assignment = inventory(tmp_path)
    output = tmp_path / "export"
    report = export(prepared, assignment, output)
    assert report["upstream_split"] == "test" and report["split_kind"] == "custom_project_split"
    assert report["training_ready"] is False
    for split in assignment:
        rows = [json.loads(line) for line in (output / f"{split}.gym.jsonl").read_text().splitlines()]
        assert len(rows) == 3 and {r["task"]["category"] for r in rows} == set(LEVELS)
        for row in rows:
            assert row["route"] == "single" and row["responses_create_params"]["tools"] == []
            assert "PRIVATE_ASSERTION" in json.dumps(row["task"]["vision_manifest"])
            policy = json.dumps(row["responses_create_params"])
            assert "PRIVATE_ASSERTION" not in policy and "PRIVATE_OBJECTIVE" not in policy
            assert "workflow.json" not in policy and "prototype" in policy
    with pytest.raises(FileExistsError):
        export(prepared, assignment, output)


@pytest.mark.parametrize("fault", ["duplicate", "missing", "unknown", "empty", "partial_level"])
def test_bad_assignments_never_create_a_partial_export(tmp_path, fault):
    prepared, assignment = inventory(tmp_path)
    if fault == "duplicate":
        assignment["validation"].append(assignment["train"][0])
    elif fault == "missing":
        assignment["train"].pop()
    elif fault == "unknown":
        assignment["train"][0] = "vision2web/webpage/missing"
    elif fault == "empty":
        assignment["train"] = []
    else:
        assignment["validation"].append(assignment["train"].pop())
    with pytest.raises(ValueError):
        export(prepared, assignment, tmp_path / "export")
    assert not (tmp_path / "export").exists()


@pytest.mark.parametrize("signal", ["image", "requirements", "project_name"])
def test_cross_split_project_variants_rejected(tmp_path, signal):
    _, assignment = inventory(tmp_path)
    train = tmp_path / "data/frontend/project-1-0"
    validation = tmp_path / "data/frontend/project-1-1"
    if signal == "image":
        (validation / "prototypes/home.jpg").write_bytes((train / "prototypes/home.jpg").read_bytes())
    elif signal == "requirements":
        (validation / "prompt.txt").write_text("Build   project-1-0.\n")
    else:
        validation.rename(validation.parent / "project-0-0")
        assignment["validation"][1] = "vision2web/frontend/project-0-0"
    prepared = tmp_path / "prepared-new"
    prepare(tmp_path / "data", prepared)
    with pytest.raises(ValueError, match="overlap"):
        export(prepared, assignment, tmp_path / "export")
    assert not (tmp_path / "export").exists()


@pytest.mark.parametrize("fault", ["manifest", "source", "budget"])
def test_changed_evidence_or_invalid_budget_is_not_exported(tmp_path, fault):
    prepared, assignment = inventory(tmp_path)
    if fault == "manifest":
        with (prepared / "website/project-2-0.json").open("a") as stream:
            stream.write(" ")
    elif fault == "source":
        (tmp_path / "data/website/project-2-0/prd.md").write_text("Changed requirements")
    with pytest.raises(ValueError):
        export(prepared, assignment, tmp_path / "export", 0 if fault == "budget" else 32768)
    assert not (tmp_path / "export").exists()
