"""Export hash-verified Manim records to the existing single-turn Gym input contract."""

import argparse
import hashlib
import json
from pathlib import Path

from mmrl.tasks import Task, write_rows


def export(prepared, output, split, max_output_tokens=32768):
    if split not in {"train", "test"}:
        raise ValueError("Use an official train or test split")
    report = json.loads((prepared / "preparation.json").read_text())
    path = prepared / f"{split}.jsonl"
    if hashlib.sha256(path.read_bytes()).hexdigest() != report["files_sha256"][path.name]:
        raise ValueError("Prepared records changed after the split/overlap audit")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    tasks = []
    for row in rows:
        row = dict(row)
        if row.pop("split") != split or row["benchmark"] != "manimbench_v1":
            raise ValueError("Unexpected benchmark or split")
        tasks.append(Task(**row))
    write_rows(tasks, output, "single", max_output_tokens=max_output_tokens)
    return {"split": split, "rows": len(tasks), "route": "single",
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "source_revision": report["revision"], "runtime_acceptance_required": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--max-output-tokens", type=int, default=32768)
    args = parser.parse_args()
    print(json.dumps(export(args.prepared, args.output, args.split, args.max_output_tokens)))


if __name__ == "__main__":
    main()
