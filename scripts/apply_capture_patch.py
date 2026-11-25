"""Apply a native integration patch only to its recorded source snapshot."""

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--rl-root", type=Path)
    source.add_argument("--gym-root", type=Path)
    source.add_argument(
        "--mamba-root", type=Path, help="Installed mamba_ssm/ops/triton directory"
    )
    parser.add_argument(
        "--check", action="store_true", help="Verify applied sources only"
    )
    parser.add_argument(
        "--process-tools", action="store_true",
        help="Apply the optional Gym MCP request-identity patch instead of the base integration",
    )
    parser.add_argument(
        "--linear-logprobs", action="store_true",
        help="Apply the optional bounded-memory AutoModel projection after the base integration",
    )
    parser.add_argument("--process-returns", action="store_true",
                        help="Apply optional verified per-call returns to the native GRPO learner")
    parser.add_argument("--local-padding", action="store_true",
                        help="Apply optional per-DP lengths after the linear-logprob integration")
    parser.add_argument("--media-custody", action="store_true",
                        help="Apply optional synchronous queue-media acceptance evidence")
    parser.add_argument("--ready-first", action="store_true", help="Apply the integrated continuous-generation runtime after base capture")
    parser.add_argument("--transport-diagnostics", action="store_true", help="Record the actual selected Gym inference backend")
    parser.add_argument("--super-multiturn", action="store_true", help="Apply deployed Super multi-turn fixes after the integrated native or Gym profiles")
    parser.add_argument("--sibling-mixed", action="store_true", help="Apply same-prompt sibling routing and recovery hooks")
    parser.add_argument("--original-history", action="store_true", help="Apply optional original tool-history capture after the Gym multi-turn profile")
    parser.add_argument("--history-bridge", action="store_true", help="Preserve original-history capture protection in native Gym configuration")
    args = parser.parse_args()
    modes = [args.process_tools, args.linear_logprobs, args.process_returns, args.local_padding, args.media_custody, args.ready_first, args.transport_diagnostics, args.super_multiturn, args.sibling_mixed, args.original_history, args.history_bridge]
    if sum(modes) > 1:
        parser.error("Apply one patch profile per invocation")
    if args.super_multiturn and not (args.rl_root or args.gym_root):
        parser.error("--super-multiturn requires --rl-root or --gym-root")
    if args.history_bridge and not args.rl_root:
        parser.error("--history-bridge requires --rl-root")
    if args.original_history and not args.gym_root:
        parser.error("--original-history requires --gym-root")
    if args.sibling_mixed and not args.rl_root:
        parser.error("--sibling-mixed requires --rl-root")
    if args.ready_first and not args.rl_root:
        parser.error("--ready-first requires --rl-root")
    if args.transport_diagnostics and not args.gym_root:
        parser.error("--transport-diagnostics requires --gym-root")

    if args.media_custody and (not args.rl_root or args.local_padding or args.linear_logprobs or args.process_returns):
        parser.error("--media-custody requires --rl-root and a separate invocation")
    if args.local_padding and (not args.rl_root or args.linear_logprobs or args.process_returns):
        parser.error("--local-padding requires --rl-root and a separate invocation")
    if args.process_tools and not args.gym_root:
        parser.error("--process-tools requires --gym-root")
    if args.linear_logprobs and not args.rl_root:
        parser.error("--linear-logprobs requires --rl-root")
    if args.process_returns and (not args.rl_root or args.linear_logprobs):
        parser.error("--process-returns requires --rl-root and a separate invocation from --linear-logprobs")
    root = (args.rl_root or args.gym_root or args.mamba_root).resolve(strict=True)
    patch_name = (
        ("native-history-bridge" if args.history_bridge else "native-sibling-mixed" if args.sibling_mixed else "native-super-multiturn" if args.super_multiturn else "native-ready-first" if args.ready_first else "native-media-custody" if args.media_custody else "automodel-local-padding" if args.local_padding else "native-process-returns" if args.process_returns else "automodel-linear-logprobs" if args.linear_logprobs else "automodel-captured-images")
        if args.rl_root
        else ("gym-original-history" if args.original_history else "gym-super-multiturn" if args.super_multiturn else "gym-transport-diagnostics" if args.transport_diagnostics else "gym-process-tools" if args.process_tools else "gym-harness-protocol")
        if args.gym_root
        else "mamba-long-sequence-indexing"
    )
    directory = Path(__file__).resolve().parents[1] / "patches"
    if args.gym_root:
        external = os.environ.get("MMRL_EXTERNAL_GYM_PATCH_DIR", "")
        if not external:
            raise SystemExit("Gym patches are not distributed here. Supply a compatible external runtime and set MMRL_EXTERNAL_GYM_PATCH_DIR for its verification manifests.")
        directory = Path(external).resolve()
    patch = directory / f"{patch_name}.patch"
    if not patch.is_file() or not (directory / f"{patch_name}.json").is_file():
        raise SystemExit(f"Missing external runtime patch or manifest: {patch_name}")
    manifest = json.loads((directory / f"{patch_name}.json").read_text())
    if digest(patch) != manifest["patch_sha256"]:
        raise SystemExit(f"{patch_name} patch does not match its manifest")
    files = manifest["files"]
    current = {name: digest(root / name) for name in files}
    if all(current[name] == spec["after_sha256"] for name, spec in files.items()):
        print(f"{patch_name} sources match; GPU acceptance is separate")
        return
    if args.check:
        raise SystemExit(f"{patch_name} patch is missing or modified")
    mismatches = [
        name for name, spec in files.items() if current[name] != spec["before_sha256"]
    ]
    if mismatches:
        raise SystemExit(
            "Refusing a different or partially patched snapshot: "
            + ", ".join(mismatches)
        )
    # NeMo's third-party workspace may be a symlink. GNU patch refuses to
    # traverse it; stage ordinary files and verify every result before install.
    with tempfile.TemporaryDirectory(prefix="mmrl-patch-") as directory:
        stage = Path(directory)
        for name in files:
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise SystemExit("Invalid patch path: " + name)
            if (root / name).exists():
                target = stage / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((root / name).read_bytes())
        command = ["patch", "--batch", "--forward", "-p1", "-i", str(patch)]
        subprocess.run(command, cwd=stage, check=True)
        for name, spec in files.items():
            if digest(stage / name) != spec["after_sha256"]:
                raise SystemExit("Staged source mismatch: " + name)
        # Refuse a source change that happened during patch preparation.
        if any(digest(root / name) != current[name] for name in files):
            raise SystemExit("Sources changed while staging the patch")
        for name in files:
            target = (root / name).resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            mode = target.stat().st_mode & 0o777 if target.exists() else 0o644
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
                stream.write((stage / name).read_bytes())
                temporary = Path(stream.name)
            os.chmod(temporary, mode)
            temporary.replace(target)
        for name, spec in files.items():
            if digest(root / name) != spec["after_sha256"]:
                raise SystemExit("Post-apply source mismatch: " + name)
    print(f"{patch_name} applied; GPU acceptance is still required")


if __name__ == "__main__":
    main()
