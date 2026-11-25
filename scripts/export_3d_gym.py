"""Export geometry-only text/image 3D tasks through the single-turn Gym contract."""

import argparse
from collections import Counter
import json
from pathlib import Path
import tempfile

from download_benchmarks import digest
from mmrl.tasks import Task, gym_row
from mmrl.vision_data import checked_file

IMAGE_PROMPT = ("Reconstruct the object shown in the following reference image(s) as a "
                "Blender 5.0 Python script. Treat all images as views of the SAME object.")


def export(prepared, output, mode, max_output_tokens=32768):
    if mode not in {"text", "image", "both"}:
        raise ValueError("Select text, image or both input modes")
    report = json.loads((prepared / "preparation.json").read_text())
    if (report.get("benchmark") != "3dcodebench" or report.get("reference_column") != "code_geo"
            or report.get("split_kind") != "custom_generator_family_split"):
        raise ValueError("Expected prepared geometry targets and a custom family split")
    for name, expected in report["files_sha256"].items():
        if digest(checked_file(prepared, name)) != expected:
            raise ValueError("Prepared code, data or preview changed")
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    counts, hashes = {}, {}
    with tempfile.TemporaryDirectory(prefix=".3d-export-", dir=output.parent) as temporary:
        work = Path(temporary)
        for split in ("train", "validation"):
            counts[split] = Counter()
            expected_families = set(report["holdout_families"] if split == "validation" else report["train_families"])
            seen, count = set(), 0
            source = checked_file(prepared, f"{split}.jsonl")
            destination = work / f"{split}.gym.jsonl"
            with source.open() as stream, destination.open("w") as target:
                for line in stream:
                    row = json.loads(line)
                    if row["split"] != split or row["family"] not in expected_families or row["task_id"] in seen:
                        raise ValueError("Prepared row violates the audited split")
                    seen.add(row["task_id"])
                    count += 1
                    for selected in (("text", "image") if mode == "both" else (mode,)):
                        preview = checked_file(prepared, row["preview"])
                        task = Task("3dcodebench", row["task_id"] + "/" + selected,
                                    row["prompt"] if selected == "text" else IMAGE_PROMPT,
                                    [] if selected == "text" else [str(preview.resolve())],
                                    category=row["family"], ground_truth=row["ground_truth"])
                        target.write(json.dumps(gym_row(task, "single", max_output_tokens)) + "\n")
                        counts[split][selected] += 1
            if count != report["counts"][split]:
                raise ValueError("Prepared split row count changed")
            hashes[destination.name] = digest(destination)
        result = {
            "benchmark": "3dcodebench", "route": "single", "input_mode": mode,
            "max_output_tokens": max_output_tokens, "counts": counts, "files_sha256": hashes,
            "source": report["source"], "prepared_manifest_sha256": digest(prepared / "preparation.json"),
            "split_kind": report["split_kind"], "upstream_split": report["upstream_split"],
            "training_ready": False,
            "note": "Reference execution, canonical multi-view Quality judge scoring and native Super admission remain required. Code and reference factories are private verifier metadata, never policy messages.",
        }
        (work / "export.json").write_text(json.dumps(result, indent=2) + "\n")
        work.rename(output)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-mode", choices=("text", "image", "both"), required=True)
    parser.add_argument("--max-output-tokens", type=int, default=32768)
    args = parser.parse_args()
    print(json.dumps(export(args.prepared, args.output, args.input_mode, args.max_output_tokens)))
