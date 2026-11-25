"""Derive Gym server requirements using the trainer's locked Ray wheel hashes."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import tomllib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gym-requirements", required=True)
    parser.add_argument("--rl-lock", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = Path(args.gym_requirements)
    lock = Path(args.rl_lock)
    packages = tomllib.loads(lock.read_text())["package"]
    matches = [package for package in packages if package["name"] == "ray"]
    if len(matches) != 1:
        raise ValueError("Expected exactly one Ray version in the training lock")
    ray = matches[0]
    hashes = sorted({wheel["hash"] for wheel in ray["wheels"]})
    if not hashes or any(not h.startswith("sha256:") for h in hashes):
        raise ValueError("Expected SHA256-locked Ray wheels")
    blocks = re.split(r"\n(?=[A-Za-z0-9][A-Za-z0-9_.-]*==)", source.read_text())
    previous = [b for b in blocks if b.startswith("ray==")]
    if len(previous) != 1:
        raise ValueError("Expected exactly one Ray block in the Gym export")
    replacement = (
        "ray=="
        + ray["version"]
        + " \\\n"
        + " \\\n".join("    --hash=" + digest for digest in hashes)
    )
    result = "\n".join(replacement if b == previous[0] else b for b in blocks)
    output = Path(args.output)
    with output.open("x") as handle:
        handle.write(result.rstrip() + "\n")
    manifest = {
        "gym_export_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "training_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
        "gym_ray": previous[0].splitlines()[0].rstrip(" \\"),
        "training_ray": ray["version"],
        "requirements_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "scope": "Gym servers sharing trainer Ray; remote harness image unchanged",
    }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
