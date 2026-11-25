"""Inventory all three extracted Vision2Web levels; do not create implicit training splits."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mmrl.vision_data import LEVELS, MissingPrototypeError, prepare_project  # noqa: E402


def prepare(data, output, quarantine_missing_prototypes=False):
    projects, quarantined = [], []
    for level in LEVELS:
        directory = data / level
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError(f"Missing official level: {level}")
        paths = sorted(directory.iterdir())
        if not paths:
            raise ValueError(f"Empty official level: {level}")
        for path in paths:
            if path.name.startswith("."):
                continue
            if path.is_symlink() or not path.is_dir():
                raise ValueError("Unexpected project entry; no silent task skipping")
            try:
                projects.append(prepare_project(path, level))
            except MissingPrototypeError as exc:
                if not quarantine_missing_prototypes:
                    raise
                quarantined.append({"task_id": f"vision2web/{level}/{path.name}",
                                    "reason": "missing_official_visual_target",
                                    "missing_asset": exc.relative,
                                    "workflow_sha256": hashlib.sha256((path / "workflow.json").read_bytes()).hexdigest()})
    # Official test tasks remain labeled as such until a deliberate project-level
    # training/holdout split is prepared and its provenance recorded separately.
    if {row["level"] for row in projects} != set(LEVELS):
        raise ValueError("Preparation must retain every official difficulty level")
    output.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for row in projects:
        path = output / row["level"] / (Path(row["project_root"]).name + ".json")
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(row, indent=2, ensure_ascii=False) + "\n")
        hashes[path.relative_to(output).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    summary = {"benchmark": "vision2web", "upstream_split": "test",
               "counts": dict(Counter(row["level"] for row in projects)),
               "files_sha256": hashes, "training_ready": False,
               "quarantined": quarantined, "discovered_projects": len(projects) + len(quarantined),
               "note": "Prepared asset/test inventory. Dataset archive provenance, custom split and native admission are separate checks."}
    (output / "preparation.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Extracted root containing webpage/frontend/website")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quarantine-missing-prototypes", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare(args.data, args.output, args.quarantine_missing_prototypes)))
