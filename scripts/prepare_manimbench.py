"""Prepare source records; rendering and Gym admission are separate requirements."""

import argparse
import hashlib
import json
from pathlib import Path

from download_benchmarks import target, verified


def prepare(train, test):
    """Keep official test rows intact; exclude exact train/test prompt or code overlap."""
    prepared = {}
    for split, rows in (("train", train), ("test", test)):
        examples = []
        for index, row in enumerate(rows):
            prompt, code = row.get("Reviewed Description"), row.get("Code")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"Missing reviewed description: {split}/{index}")
            if not isinstance(code, str) or not code.strip():
                raise ValueError(f"Missing reference code: {split}/{index}")
            if row.get("Split") != split:
                raise ValueError(f"Declared split mismatch: {split}/{index}")
            examples.append({
                "task_id": f"manimbench-v1/{split}/{index:04d}",
                "benchmark": "manimbench_v1",
                "split": split,
                "prompt": prompt,
                "category": row.get("Type", ""),
                "references": [],
                "ground_truth": code,
            })
        prepared[split] = examples
    test_prompts = {r["prompt"].strip() for r in prepared["test"]}
    test_codes = {r["ground_truth"].strip() for r in prepared["test"]}
    kept, excluded = [], []
    for row in prepared["train"]:
        reasons = []
        if row["prompt"].strip() in test_prompts:
            reasons.append("exact_test_prompt")
        if row["ground_truth"].strip() in test_codes:
            reasons.append("exact_test_code")
        if reasons:
            excluded.append({"task_id": row["task_id"], "reasons": reasons})
        else:
            kept.append(row)
    prepared["train"] = kept
    return prepared, excluded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lock = json.loads((Path(__file__).resolve().parents[1] / "benchmarks/sources.lock.json").read_text())
    source = lock["datasets"]["manimbench"]
    for artifact in source["artifacts"]:
        if not verified(target(args.data, artifact["destination"]), artifact):
            raise ValueError("Source data does not match pinned ManimBench revision")
    import pyarrow.parquet as pq

    raw = {s: pq.read_table(args.data / "manimbench" / f"manim_sft_dataset_{s}_v2.parquet").to_pylist()
           for s in ("train", "test")}
    prepared, excluded = prepare(raw["train"], raw["test"])
    args.output.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for split, rows in prepared.items():
        path = args.output / (split + ".jsonl")
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    report = {"repository": source["repository"], "revision": source["revision"],
              "license": source["license_note"], "input_counts": {k: len(v) for k, v in raw.items()},
              "output_counts": {k: len(v) for k, v in prepared.items()},
              "excluded_training_rows": excluded, "files_sha256": hashes,
              "rendering_verified": False, "training_ready": False,
              "note": "Ground truth is private verifier data, never part of model input."}
    (args.output / "preparation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
