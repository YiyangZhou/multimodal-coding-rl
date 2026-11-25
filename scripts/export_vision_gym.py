"""Export an explicit custom Vision2Web split; preserve upstream test provenance."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from mmrl.tasks import Task, gym_row
from mmrl.vision_data import LEVELS, checked_file, verify_manifest


def load_projects(prepared):
    report = json.loads((prepared / "preparation.json").read_text())
    if report.get("benchmark") != "vision2web" or report.get("upstream_split") != "test":
        raise ValueError("Expected the prepared Vision2Web test inventory")
    projects = {}
    for relative, expected in report["files_sha256"].items():
        path = checked_file(prepared, relative)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Prepared project manifest changed")
        manifest = verify_manifest(json.loads(path.read_text()))
        if manifest["task_id"] in projects:
            raise ValueError("Duplicate project identity")
        projects[manifest["task_id"]] = manifest
    counts = dict(Counter(m["level"] for m in projects.values()))
    if counts != report["counts"] or set(counts) != set(LEVELS):
        raise ValueError("Incomplete three-level project inventory")
    return projects


def validate_assignment(projects, assignment):
    """Require complete project assignment; detect exact cross-split overlap."""
    if not isinstance(assignment, dict) or set(assignment) != {"train", "validation"}:
        raise ValueError("Provide explicit custom train and validation project lists")
    seen, signals = {}, {}
    for split, ids in assignment.items():
        if not isinstance(ids, list) or not ids or any(not isinstance(i, str) for i in ids):
            raise ValueError("Both custom splits need nonempty project ID lists")
        for task_id in ids:
            if task_id not in projects or task_id in seen:
                raise ValueError("Unknown, duplicate or overlapping project assignment")
            seen[task_id] = split
            manifest = projects[task_id]
            # A project may appear at multiple difficulty levels. Identical
            # prototype files also cannot be used on opposite sides of a split.
            keys = [("project_name", Path(manifest["project_root"]).name)]
            keys += [("prototype", item["sha256"]) for item in manifest["public_assets"]
                     if item["path"].startswith("prototypes/")]
            # All webpage tasks share a generic instruction; it is not identity.
            if manifest["level"] != "webpage":
                keys.append(("requirements", " ".join(manifest["requirements"].split())))
            for key in keys:
                if key in signals and signals[key] != split:
                    raise ValueError("Project, exact prototype or requirement overlap across custom splits")
                signals[key] = split
        if {projects[i]["level"] for i in ids} != set(LEVELS):
            raise ValueError("Both custom splits must cover all three Vision2Web levels")
    if set(seen) != set(projects):
        raise ValueError("Every prepared project must be assigned exactly once")


def export(prepared, assignment, output, max_output_tokens=32768):
    projects = load_projects(prepared)
    validate_assignment(projects, assignment)
    # Construct and validate every policy input before creating output artifacts.
    encoded, counts = {}, {}
    for split, ids in assignment.items():
        rows = []
        for task_id in sorted(ids):
            manifest = projects[task_id]
            root = Path(manifest["project_root"])
            task = Task(benchmark="vision2web", task_id=task_id,
                        prompt=manifest["requirements"], category=manifest["level"],
                        references=[str(root / a["path"]) for a in manifest["public_assets"]
                                    if a["path"].startswith("prototypes/")],
                        vision_manifest=manifest)
            rows.append(json.dumps(gym_row(task, "single", max_output_tokens)))
        encoded[split] = "\n".join(rows) + "\n"
        counts[split] = dict(Counter(projects[i]["level"] for i in ids))
    output.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for split, content in encoded.items():
        path = output / f"{split}.gym.jsonl"
        path.write_text(content)
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    report = {
        "benchmark": "vision2web", "upstream_split": "test", "split_kind": "custom_project_split",
        "assignment": assignment, "counts": counts, "files_sha256": hashes,
        "prepared_inventory_sha256": hashlib.sha256((prepared / "preparation.json").read_bytes()).hexdigest(),
        "route": "single", "max_output_tokens": max_output_tokens, "training_ready": False,
        "limitations": "Exact project/image/requirement overlap checked; perceptual or semantic near-duplicates require review. Official archive provenance and native runtime admission remain required. Custom validation is not the official benchmark score.",
    }
    (output / "export.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--assignment", type=Path, required=True,
                        help='JSON with explicit {"train": [task IDs], "validation": [task IDs]}')
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-output-tokens", type=int, default=32768)
    args = parser.parse_args()
    print(json.dumps(export(args.prepared, json.loads(args.assignment.read_text()), args.output,
                            args.max_output_tokens)))
