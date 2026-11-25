"""Bind synchronous media receipts to durable capture lineage after a job."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(receipts, lineage, *, job, observer_sha256):
    ledger, calls, rows, evidence = {}, {}, {}, {}
    for path in sorted(Path(lineage).glob("*.lineage.jsonl")):
        evidence[str(path)] = sha(path)
        for line in path.read_text().splitlines():
            value = json.loads(line)
            if "staging_key" not in value or "failure_reason" in value:
                continue
            key = value["staging_key"]
            if key in ledger and ledger[key] != value:
                raise ValueError("Conflicting durable lineage")
            ledger[key] = value
    paths = sorted(Path(receipts).glob("*.json"))
    if not paths:
        raise ValueError("No synchronous media receipts")
    for path in paths:
        value = json.loads(path.read_text())
        if (value.get("schema_version") != 1 or value.get("job") != job or
                value.get("observer_sha256") != observer_sha256 or
                value.get("observation_boundary") != "published_queue_before_finalizer_returns"):
            raise ValueError("Wrong media receipt provenance")
        evidence[str(path)] = sha(path)
        for key, call in value["calls"].items():
            recorded = ledger[key]
            for name in ("staging_digest", "model_call_id", "parent_call_id", "weight_version", "prev_len", "cum_len"):
                if call[name] != recorded.get(name):
                    raise ValueError("Media evidence differs from capture lineage: " + name)
            if key in calls and calls[key] != call:
                raise ValueError("Conflicting observed call")
            calls[key] = call
        for sample, observed in value["rows"].items():
            if sample in rows:
                raise ValueError("Duplicate canonical sample across finalized groups")
            terminal = observed["terminal_call"]
            if ledger[terminal]["cumulative_hash"] != observed["token_hash"]:
                raise ValueError("Media terminal tokens differ from capture lineage")
            chain, current, seen = [], terminal, set()
            while current is not None:
                if current in seen or current.rsplit("/", 1)[0] != observed["rollout_id"]:
                    raise ValueError("Invalid observed media chain")
                seen.add(current)
                call = value["calls"][current]
                chain.append(call)
                parent = call["parent_call_id"]
                current = observed["rollout_id"] + "/" + parent if parent else None
            if (len(chain) != observed["captured_calls"] or
                    sum(c["media_present"] for c in chain) != observed["media_calls"]):
                raise ValueError("Incomplete observed media chain")
            rows[sample] = observed
    if not rows:
        raise ValueError("No valid observed canonical rows")
    common = {"job": job, "evidence_sha256": evidence,
              "observer_sha256": observer_sha256,
              "learner_kernel_image_tensors_independently_verified": False,
              "scope": "Synchronous staged-to-published queue readback, bound to durable lineage; learner dump binding remains separate."}
    return common | {"calls": calls}, common | {"rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--job", type=int, required=True)
    parser.add_argument("--observer", type=Path, required=True)
    args = parser.parse_args()
    stage, canonical = summarize(args.run / "media/receipts", args.run / "captures/lineage",
                                 job=args.job, observer_sha256=sha(args.observer))
    for name, value in (("staged-media-fingerprints.json", stage),
                        ("canonical-media-comparison.json", canonical)):
        with (args.run / "media" / name).open("x") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")
    print(json.dumps({"job": args.job, "rows": len(canonical["rows"]), "calls": len(stage["calls"])}))


if __name__ == "__main__":
    main()
