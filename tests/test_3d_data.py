import copy
import io
import json
from pathlib import Path
import runpy
import sys

from PIL import Image
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
prepare = runpy.run_path(str(SCRIPTS / "prepare_3dcodebench.py"))["prepare"]
export = runpy.run_path(str(SCRIPTS / "export_3d_gym.py"))["export"]


def rows():
    result = []
    for index, family in enumerate(("Chair", "Beetle")):
        for seed in range(2):
            name = f"{family}_{seed:03d}"
            image = io.BytesIO()
            Image.new("RGB", (16, 16), (index * 100, seed * 100, 80)).save(image, format="PNG")
            result.append({"factory": name, "preview": {"bytes": image.getvalue(), "path": "ignored.png"},
                           "caption_object": f"Description of {name}",
                           "caption_create_3D_object": f"Create {name}",
                           "code": f"PRIVATE_MATERIAL_{name}", "code_geo": f"PRIVATE_GEOMETRY_{name}"})
    return result


def build(tmp_path, data=None):
    output = tmp_path / "prepared"
    report = prepare(iter(data if data is not None else rows()), output,
                     {"Chair", "Beetle"}, {"Beetle"}, {"fixture": True}, seeds=2)
    return output, report


@pytest.mark.parametrize("mode", ["text", "image", "both"])
def test_private_geometry_target_and_family_split_survive_gym_export(tmp_path, mode):
    prepared, report = build(tmp_path)
    assert report["counts"] == {"train": 2, "validation": 2}
    assert report["training_ready"] is False
    source = json.loads((prepared / "train.jsonl").read_text().splitlines()[0])
    assert source["ground_truth"].startswith("PRIVATE_GEOMETRY_")
    assert source["full_material_code"].startswith("PRIVATE_MATERIAL_")
    output = tmp_path / "export"
    result = export(prepared, output, mode)
    assert result["training_ready"] is False
    ids = {}
    for split, family in (("train", "Chair"), ("validation", "Beetle")):
        records = [json.loads(line) for line in (output / f"{split}.gym.jsonl").read_text().splitlines()]
        assert len(records) == (4 if mode == "both" else 2)
        ids[split] = {r["task"]["task_id"] for r in records}
        for record in records:
            assert record["task"]["category"] == family
            assert record["task"]["ground_truth"].startswith("PRIVATE_GEOMETRY_")
            policy = record["responses_create_params"]
            assert "PRIVATE_" not in json.dumps(policy)
            assert record["route"] == "single" and policy["tools"] == []
            images = [x for x in policy["input"][-1]["content"] if x["type"] == "input_image"]
            assert len(images) == (1 if record["task"]["task_id"].endswith("/image") else 0)
    assert not ids["train"] & ids["validation"]
    with pytest.raises(FileExistsError):
        export(prepared, output, mode)


@pytest.mark.parametrize("fault", ["missing", "duplicate", "unknown_family", "bad_seed", "missing_geometry", "external_preview"])
def test_bad_source_never_leaves_a_prepared_dataset(tmp_path, fault):
    data = rows()
    if fault == "missing":
        data.pop()
    elif fault == "duplicate":
        data.append(copy.deepcopy(data[0]))
    elif fault == "unknown_family":
        data[0]["factory"] = "Unknown_000"
    elif fault == "bad_seed":
        data[0]["factory"] = "Chair_999"
    elif fault == "missing_geometry":
        data[0]["code_geo"] = ""
    else:
        data[0]["preview"] = {"bytes": None, "path": "/private/reference.png"}
    with pytest.raises(ValueError):
        build(tmp_path, data)
    assert not (tmp_path / "prepared").exists()
    assert not list(tmp_path.glob(".3d-prepare-*"))


@pytest.mark.parametrize("field", ["preview", "code_geo", "caption_create_3D_object"])
def test_exact_overlap_across_families_requires_revised_split(tmp_path, field):
    data = rows()
    data[2][field] = copy.deepcopy(data[0][field])
    with pytest.raises(ValueError, match="overlap"):
        build(tmp_path, data)
    assert not (tmp_path / "prepared").exists()


@pytest.mark.parametrize("holdout", [set(), {"Unknown"}, {"Chair", "Beetle"}])
def test_family_partition_must_be_nonempty_and_disjoint(tmp_path, holdout):
    with pytest.raises(ValueError):
        prepare(rows(), tmp_path / "prepared", {"Chair", "Beetle"}, holdout, {}, seeds=2)


@pytest.mark.parametrize("changed", ["train.jsonl", "previews/Chair_000.png"])
def test_changed_prepared_evidence_cannot_enter_gym(tmp_path, changed):
    prepared, _ = build(tmp_path)
    with (prepared / changed).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="changed"):
        export(prepared, tmp_path / "export", "both")
    assert not (tmp_path / "export").exists()


def test_invalid_output_budget_leaves_no_partial_export(tmp_path):
    prepared, _ = build(tmp_path)
    with pytest.raises(ValueError, match="budget"):
        export(prepared, tmp_path / "export", "both", 0)
    assert not (tmp_path / "export").exists()


def test_base_factory_and_seed_variants_share_one_split(tmp_path):
    data = rows()
    base = copy.deepcopy(data[0])
    base.update(factory="ChairFactory", caption_create_3D_object="Base chair",
                code_geo="BASE_PRIVATE_GEOMETRY")
    data.append(base)
    report = prepare(data, tmp_path / "prepared", {"Chair", "Beetle"}, {"Chair"}, {},
                     seeds=2, base_families={"Chair"})
    assert report["counts"] == {"validation": 3, "train": 2}
    val = [json.loads(x) for x in (tmp_path / "prepared/validation.jsonl").read_text().splitlines()]
    assert {x["factory"] for x in val} == {"ChairFactory", "Chair_000", "Chair_001"}
    assert next(x for x in val if x["factory"] == "ChairFactory")["seed"] is None
    assert report["base_families"] == ["Chair"]


def test_missing_expected_base_factory_is_not_silently_ignored(tmp_path):
    with pytest.raises(ValueError, match="Incomplete"):
        prepare(rows(), tmp_path / "prepared", {"Chair", "Beetle"}, {"Chair"}, {},
                seeds=2, base_families={"Chair"})
    assert not (tmp_path / "prepared").exists()


def test_relative_previews_require_pinned_bytes(tmp_path):
    import hashlib
    preview_bytes = runpy.run_path(str(SCRIPTS / "prepare_3dcodebench.py"))["preview_bytes"]
    name = "ChairFactory/renders/Image_005.webp"
    image = rows()[0]["preview"]["bytes"]
    destination = "3dcodebench/3DCodeData/" + name
    file = tmp_path / destination
    file.parent.mkdir(parents=True)
    file.write_bytes(image)
    artifact = {"destination": destination, "size": len(image), "hash_algorithm": "sha256",
                "hash": hashlib.sha256(image).hexdigest()}
    pins = {"3DCodeData/" + name: artifact}
    assert preview_bytes(name, tmp_path, pins)[1] == artifact["hash"]
    with pytest.raises(ValueError, match="pinned"):
        preview_bytes(name)
    file.write_bytes(b"changed")
    with pytest.raises(ValueError, match="pinned"):
        preview_bytes(name, tmp_path, pins)
