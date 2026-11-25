"""Prepare pinned 3DCodeData with an explicit family-disjoint custom split."""

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
import hashlib
import io
import json
from pathlib import Path
import re
import tempfile

from PIL import Image

from download_benchmarks import digest, target, verified

FIELDS = ("factory", "preview", "caption_object", "caption_create_3D_object", "code", "code_geo")


def instance_identity(name):
    if not isinstance(name, str):
        raise ValueError("Missing 3DCodeData factory identity")
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*Factory", name):
        return name.removesuffix("Factory"), None
    match = re.fullmatch(r"([A-Za-z][A-Za-z0-9_]*)_([0-9]{3})", name)
    if not match:
        raise ValueError("Expected official <family>_<NNN> instance identity")
    return match[1], int(match[2])


def preview_bytes(value, data_root=None, preview_artifacts=None):
    # Actual parquet stores relative preview paths. Only separately pinned,
    # hash-verified images may resolve them; never follow arbitrary cell URLs.
    if isinstance(value, str):
        artifact = (preview_artifacts or {}).get("3DCodeData/" + value)
        if data_root is None or artifact is None:
            raise ValueError("Preview path lacks pinned acquisition metadata")
        path = target(data_root, artifact["destination"])
        if not verified(path, artifact):
            raise ValueError("Missing or changed pinned preview image")
        data = path.read_bytes()
    else:
        data = value.get("bytes") if isinstance(value, dict) else value
    if not isinstance(data, bytes) or not 0 < len(data) <= 32 * 1024 * 1024:
        raise ValueError("Expected embedded preview image bytes; external previews need separate pinned acquisition")
    with Image.open(io.BytesIO(data)) as image:
        if image.format not in {"PNG", "JPEG", "WEBP"} or getattr(image, "n_frames", 1) != 1:
            raise ValueError("Expected a static PNG/JPEG/WebP object preview")
        image.load()
        pixels = image.convert("RGB")
        pixel_hash = hashlib.sha256(str(pixels.size).encode() + pixels.tobytes()).hexdigest()
        output = io.BytesIO()
        pixels.save(output, format="PNG")
    return output.getvalue(), hashlib.sha256(data).hexdigest(), pixel_hash


def prepare(rows, output, families, holdout, source, seeds=60, *,
            base_families=(), data_root=None, preview_artifacts=None):
    seeded_families, base_families = set(families), set(base_families)
    families, holdout = seeded_families | base_families, set(holdout)
    if not holdout or not holdout < families:
        raise ValueError("Select a nonempty proper subset of official generator families for validation")
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    identities, overlap, counts = defaultdict(set), {}, Counter()
    hashes = {}
    with tempfile.TemporaryDirectory(prefix=".3d-prepare-", dir=output.parent) as temporary:
        work = Path(temporary)
        (work / "previews").mkdir()
        with ExitStack() as stack:
            streams = {s: stack.enter_context((work / f"{s}.jsonl").open("w"))
                       for s in ("train", "validation")}
            for row in rows:
                if set(FIELDS) - row.keys():
                    raise ValueError("3DCodeData columns differ from the pinned dataset card")
                family, seed = instance_identity(row["factory"])
                valid = (family in base_families if seed is None
                         else family in seeded_families and seed in range(seeds))
                if not valid or seed in identities[family]:
                    raise ValueError("Unknown family, duplicate instance or unexpected seed")
                identities[family].add(seed)
                for name in ("caption_object", "caption_create_3D_object", "code", "code_geo"):
                    if not isinstance(row[name], str) or not row[name].strip():
                        raise ValueError(f"Missing source field: {row['factory']}/{name}")
                png, raw_hash, pixel_hash = preview_bytes(row["preview"], data_root, preview_artifacts)
                split = "validation" if family in holdout else "train"
                # Exact duplicates across families must not silently cross the
                # split. Retain the source unchanged and ask for a revised split.
                keys = [("preview_pixels", pixel_hash),
                        ("geometry", hashlib.sha256(row["code_geo"].strip().encode()).hexdigest()),
                        ("instruction", " ".join(row["caption_create_3D_object"].split()))]
                for key in keys:
                    if key in overlap and overlap[key] != split:
                        raise ValueError("Exact preview, geometry or instruction overlap across families; revise the split")
                    overlap[key] = split
                image = Path("previews") / (row["factory"] + ".png")
                (work / image).write_bytes(png)
                hashes[image.as_posix()] = hashlib.sha256(png).hexdigest()
                record = {
                    "benchmark": "3dcodebench", "task_id": "3dcodedata/" + row["factory"],
                    "factory": row["factory"], "family": family, "seed": seed,
                    "split": split, "upstream_split": "train",
                    "prompt": row["caption_create_3D_object"], "description": row["caption_object"],
                    "ground_truth": row["code_geo"], "full_material_code": row["code"],
                    "preview": image.as_posix(), "original_preview_sha256": raw_hash,
                    "preview_pixels_sha256": pixel_hash,
                }
                streams[split].write(json.dumps(record, ensure_ascii=False) + "\n")
                counts[split] += 1
        expected = {family: (set(range(seeds)) if family in seeded_families else set())
                    | ({None} if family in base_families else set()) for family in families}
        if dict(identities) != expected:
            raise ValueError("Incomplete generator-family or seed coverage")
        for split in streams:
            hashes[f"{split}.jsonl"] = digest(work / f"{split}.jsonl")
        report = {
            "benchmark": "3dcodebench", "source": source, "upstream_split": "train",
            "split_kind": "custom_generator_family_split", "holdout_families": sorted(holdout),
            "train_families": sorted(families - holdout), "seeds_per_family": seeds,
            "seeded_families": sorted(seeded_families), "base_families": sorted(base_families),
            "counts": dict(counts), "files_sha256": hashes,
            "reference_column": "code_geo", "training_ready": False,
            "limitations": "Reference scripts have not been executed. Official 212-category evaluation overlaps these families; only held-out families support an unseen-family claim. Exact overlap checks do not prove semantic independence.",
        }
        (work / "preparation.json").write_text(json.dumps(report, indent=2) + "\n")
        # A malformed or incomplete source never leaves an apparently prepared dataset.
        work.rename(output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--holdout-families", type=Path, required=True, help="JSON list of generator family names")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview-manifest", type=Path, required=True)
    args = parser.parse_args()
    lock = json.loads((Path(__file__).resolve().parents[1] / "benchmarks/sources.lock.json").read_text())
    source = lock["datasets"]["3dcodebench"]
    artifact = next(a for a in source["artifacts"] if a["path"] == "3DCodeData/data/train.parquet")
    path = target(args.data, artifact["destination"])
    if not verified(path, artifact):
        raise ValueError("3DCodeData parquet does not match its pinned source")
    families = {Path(a["path"]).parts[1].removesuffix("_seed0")
                for a in source["artifacts"] if a["path"].startswith("3DCodeBench/")}
    holdout = json.loads(args.holdout_families.read_text())
    if (not isinstance(holdout, list) or any(not isinstance(v, str) for v in holdout)
            or len(holdout) != len(set(holdout))):
        raise ValueError("Expected a unique list of held-out family names")
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    names = parquet.read(columns=["factory"]).column(0).to_pylist()
    if len(names) != len(set(names)) or len(names) != 12963:
        raise ValueError("Pinned official corpus identities changed")
    base_families = {instance_identity(name)[0] for name in names if name.endswith("Factory")}
    if len(base_families) != 243:
        raise ValueError("Pinned official base-factory count changed")
    previews = json.loads(args.preview_manifest.read_text())["datasets"]["3dcodebench_previews"]
    if (previews["revision"] != source["revision"] or previews["repository"] != source["repository"]
            or previews["derived_from_parquet_sha256"] != artifact["hash"]):
        raise ValueError("Preview manifest does not match the pinned parquet")
    images = {item["path"]: item for item in previews["artifacts"]}
    expected_paths = {"3DCodeData/" + name + "/renders/Image_005.webp" for name in names}
    if len(images) != len(previews["artifacts"]) or set(images) != expected_paths:
        raise ValueError("Incomplete or duplicated preview manifest")
    rows = (row for batch in parquet.iter_batches(batch_size=32, columns=list(FIELDS))
            for row in batch.to_pylist())
    report = prepare(rows, args.output, families, holdout,
                     {"repository": source["repository"], "revision": source["revision"],
                      "artifact": artifact, "license": source["license_note"],
                      "preview_manifest_sha256": digest(args.preview_manifest)},
                     base_families=base_families, data_root=args.data, preview_artifacts=images)
    print(json.dumps({k: v for k, v in report.items() if k != "files_sha256"}))


if __name__ == "__main__":
    main()
