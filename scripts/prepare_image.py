"""Build a private CPU-image context from verified source snapshots and locks."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_source_pins(path):
    pins = json.loads(path.read_text())
    if not isinstance(pins, dict) or not pins:
        raise ValueError("External source lock must be a nonempty path-to-SHA256 mapping")
    for name, value in pins.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or not re.fullmatch(r"[0-9a-f]{64}", str(value)):
            raise ValueError("Invalid external source lock entry")
    return pins


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gym-root", required=True)
    p.add_argument("--hermes-root", required=True)
    p.add_argument("--gym-source-lock", required=True, type=Path,
                   help="External JSON mapping of Gym relative paths to SHA256 hashes")
    p.add_argument("--output", required=True)
    p.add_argument("--uv", default="uv")
    args = p.parse_args()
    repo = Path(__file__).resolve().parents[1]
    pins = json.loads((repo / "source.lock.json").read_text())
    gym, hermes = Path(args.gym_root).resolve(), Path(args.hermes_root).resolve()
    for root, files in (
        (gym, load_source_pins(args.gym_source_lock)),
        (hermes, pins["hermes_dependency_locks"]),
    ):
        for name, digest in files.items():
            if sha(root / name) != digest:
                raise ValueError("Source pin mismatch: " + str(root / name))
    revision = subprocess.check_output(
        ["git", "-C", str(hermes), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != pins["harness_pins"]["hermes"]:
        raise ValueError("Hermes checkout does not match the pinned revision")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    for key in (
        "UV_CONSTRAINT",
        "UV_BUILD_CONSTRAINT",
        "UV_OVERRIDE",
        "PIP_CONSTRAINT",
    ):
        env.pop(key, None)

    def uv(*command):
        subprocess.run([args.uv, "--no-config", *command], env=env, check=True)

    for root, name, extras in (
        (gym, "gym", ["--extra", "dev", "--extra", "sandbox"]),
        (hermes, "hermes", ["--no-dev"]),
    ):
        uv(
            "export",
            "--frozen",
            "--no-emit-project",
            "--project",
            str(root),
            *extras,
            "--output-file",
            str(output / f"{name}.requirements.txt"),
        )
    # Only packages missing from Hermes are added from the Gym lock. Conflicting
    # shared versions stay in their respective environment; acceptance tests are mandatory.
    hermes_names = set(
        re.findall(
            r"^([A-Za-z0-9_.-]+)==",
            (output / "hermes.requirements.txt").read_text(),
            re.M,
        )
    )
    blocks = re.split(
        r"\n(?=[A-Za-z0-9][A-Za-z0-9_.-]*==)",
        (output / "gym.requirements.txt").read_text(),
    )
    supplement = [
        b
        for b in blocks
        if (m := re.match(r"([A-Za-z0-9_.-]+)==", b)) and m[1] not in hermes_names
    ]
    (output / "hermes-gym.requirements.txt").write_text("\n".join(supplement) + "\n")
    # Hermes's broad OpenAI constraint permits Gym's newer SDK, which is
    # required by Gym's Responses schema. Keep the original export as evidence.
    bridges = {
        name: next(b for b in blocks if b.startswith(name + "=="))
        for name in ("openai", "fsspec")
    }
    hermes_blocks = re.split(
        r"\n(?=[A-Za-z0-9][A-Za-z0-9_.-]*==)",
        (output / "hermes.requirements.txt").read_text(),
    )
    runtime = [bridges.get(b.split("==", 1)[0], b) for b in hermes_blocks]
    (output / "hermes-runtime.requirements.txt").write_text("\n".join(runtime) + "\n")
    # The pinned Hermes source declares exa-py, but its existing lock predates
    # that declaration. Resolve this explicitly in the derived sandbox lock.
    (output / "hermes-extra.in").write_text("exa-py==2.9.0\n")
    uv(
        "pip",
        "compile",
        str(output / "hermes-extra.in"),
        "--constraint",
        str(output / "hermes-runtime.requirements.txt"),
        "--python-version",
        "3.13",
        "--python-platform",
        "x86_64-unknown-linux-gnu",
        "--generate-hashes",
        "--output-file",
        str(output / "hermes-extra.requirements.txt"),
    )
    (output / "renderer.in").write_text(
        "cairosvg==2.8.2\nplaywright==1.55.0\nmatplotlib==3.11.0\n"
        "numpy==2.3.2\npillow==12.3.0\n"
        "opencv-python-headless==4.13.0.92\nscikit-image==0.26.0\n"
    )
    uv(
        "pip",
        "compile",
        str(output / "renderer.in"),
        "--constraint",
        str(output / "gym.requirements.txt"),
        "--python-version",
        "3.13",
        "--python-platform",
        "x86_64-unknown-linux-gnu",
        "--generate-hashes",
        "--output-file",
        str(output / "renderer.requirements.txt"),
    )
    for name in ("nemo_gym", "responses_api_agents"):
        shutil.copytree(
            gym / name,
            output / "gym-source" / name,
            ignore=shutil.ignore_patterns(
                "__pycache__",
                "*.pyc",
                ".venv",
                "node_modules",
                "data",
                "outputs",
                "tests",
                ".env*",
            ),
        )
    # Copy tracked source only: no credentials, local caches or user trajectories.
    tracked = (
        subprocess.check_output(["git", "-C", str(hermes), "ls-files", "-z"])
        .decode()
        .split("\0")
    )
    for name in filter(None, tracked):
        source = hermes / name
        if source.is_symlink() or not source.is_file():
            continue
        target = output / "hermes-source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    shutil.copytree(
        repo / "mmrl",
        output / "mmrl",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copyfile(repo / "containers/Dockerfile.runtime", output / "Dockerfile")
    for name in ("blender.lock.json", "manim.assets.lock.json",
                 "manim.hashed.requirements.txt", "install_benchmark_renderers.py"):
        shutil.copyfile(repo / "containers" / name, output / name)
    manifest = {
        str(path.relative_to(output)): sha(path)
        for path in output.rglob("*")
        if path.is_file()
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        "Prepared image context; image build and remote acceptance still required:",
        output,
    )


if __name__ == "__main__":
    main()
