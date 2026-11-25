"""Download pinned public benchmark artifacts; never execute their contents."""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import socket
import urllib.request


def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    if algorithm == "sha1":
        h.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def target(root, relative):
    relative = PurePosixPath(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Artifact path must remain inside the dataset directory")
    root = root.resolve()
    path = root.joinpath(*relative.parts)
    if not path.resolve().is_relative_to(root):
        raise ValueError("Artifact path escapes through a symlink")
    return path


def verified(path, artifact):
    return (path.is_file() and path.stat().st_size == artifact["size"]
            and digest(path, artifact["hash_algorithm"]) == artifact["hash"])


def download(root, artifact):
    path = target(root, artifact["destination"])
    if verified(path, artifact):
        return "verified_existing"
    if path.exists():
        raise ValueError(f"Existing file failed integrity check: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    # A partial file is never admitted. A retry downloads it again from byte zero.
    with urllib.request.urlopen(artifact["url"], timeout=120) as response:
        with temporary.open("wb") as output:
            total = 0
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > artifact["size"]:
                    raise ValueError("Response exceeds pinned artifact size")
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
    if not verified(temporary, artifact):
        raise ValueError(f"Downloaded artifact failed integrity check: {path}")
    temporary.replace(path)
    return "downloaded_verified"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parents[1]
                        / "benchmarks/sources.lock.json")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--benchmark", action="append")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    names = args.benchmark or list(manifest["datasets"])
    unknown = set(names) - manifest["datasets"].keys()
    if unknown:
        parser.error(f"Unknown benchmarks: {sorted(unknown)}")
    artifacts = [a for name in names for a in manifest["datasets"][name]["artifacts"]]
    pending = [a for a in artifacts if not verified(target(args.output, a["destination"]), a)]
    if args.verify_only:
        if pending:
            raise SystemExit(f"{len(pending)} missing or unverified artifacts")
        print(json.dumps({"verified": len(artifacts)}))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "download-receipts.jsonl").open("a") as receipt:
        for artifact in artifacts:
            status = download(args.output, artifact)
            row = {"artifact": artifact["destination"], "status": status,
                   "hash": artifact["hash"], "hash_algorithm": artifact["hash_algorithm"]}
            receipt.write(json.dumps(row) + "\n")
            receipt.flush()
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
