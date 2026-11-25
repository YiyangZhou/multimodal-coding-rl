"""Validate and enter the pinned native V2 launcher; never resolve dependencies."""

import argparse
import importlib.util
import importlib.metadata
import json
import os
import sys
import subprocess
from pathlib import Path

from mmrl.sandbox import provider_config
from mmrl.config import validate_capture_retries, validate_thinking_parser, validate_sibling_recipe
from mmrl.history_config import original_history_root


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--project-harness-adapter", action="store_true", help="Use the mm_harness component for agent rollouts")
    p.add_argument("--rl-root", required=True)
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--public-runtime", action="store_true", help="Verify the experimental public-source composite runtime")
    args = p.parse_args()
    root = Path(args.rl_root).resolve()
    cfg = json.loads(Path(args.config).read_text())
    validate_thinking_parser(cfg)
    validate_capture_retries(cfg)
    installer = str(Path(__file__).with_name("apply_capture_patch.py"))
    if args.public_runtime:
        subprocess.run([sys.executable, installer, "--rl-root", str(root),
                        "--public-runtime", "--check"], check=True)
    history_enabled = original_history_root(cfg, os.environ) is not None
    if history_enabled:
        installer = str(Path(__file__).with_name("apply_capture_patch.py"))
        for source, path, profile in (
            ("--rl-root", str(root), "--sibling-mixed"),
            ("--rl-root", str(root), "--history-bridge"),
            ("--gym-root", os.environ["NEMO_GYM_ROOT"], "--original-history"),
        ):
            if args.public_runtime and source == "--rl-root":
                continue  # The composite manifest verifies these native layers together.
            subprocess.run([sys.executable, installer, source, path, profile, "--check"], check=True)
    if cfg.get("data", {}).get("train", {}).get("dataset_name") == "mmrl.mixed_dataset.BalancedMixedGymDataset":
        validate_sibling_recipe(cfg)
        if not args.public_runtime:
            subprocess.run([sys.executable, installer, "--rl-root", str(root),
                            "--sibling-mixed", "--check"], check=True)
    provider_config()  # Fail before spawning actors or accepting local defaults.
    if len(os.environ.get("MMRL_CAPTURE_GATEWAY_TOKEN", "")) < 32:
        raise SystemExit("Configure a dedicated random MMRL_CAPTURE_GATEWAY_TOKEN")
    gym = cfg["env"]["nemo_gym"]
    generation = cfg["policy"]["generation"]
    expected_sampling = {
        "temperature": generation["temperature"],
        "top_p": generation["top_p"],
        "top_k": -1,
    }
    model = gym["policy_model"]["responses_api_models"]["vllm_model"]
    if (
        generation["top_k"] not in (None, -1)
        or model.get("sampling_overrides") != expected_sampling
    ):
        raise SystemExit(
            "Recompose the recipe: Gym sampling must match the training generation profile"
        )
    checked = set()
    for component in (
        "resources_servers/mm_coding",
        "responses_api_agents/simple_agent",
        ("responses_api_agents/mm_harness" if args.project_harness_adapter
         else "responses_api_agents/harness_agent"),
        "responses_api_models/vllm_model",
    ):
        venv = Path(gym["uv_venv_dir"]) / component / ".venv"
        if not (venv / "bin/python").is_file() or not (venv / "bin/activate").is_file():
            raise SystemExit(
                "Bind the prevalidated Gym environment first: " + component
            )
        if venv.resolve() not in checked:
            if history_enabled:
                subprocess.run([
                    str(venv / "bin/python"),
                    str(Path(__file__).with_name("check_history_imports.py")),
                ], check=True)
            runtime = json.loads(
                subprocess.check_output(
                    [
                        str(venv / "bin/python"),
                        "-c",
                        "import sys,json,importlib.metadata; "
                        "print(json.dumps({'ray':importlib.metadata.version('ray'),"
                        "'python':list(sys.version_info[:3])}))",
                    ],
                    text=True,
                )
            )
            if runtime["ray"] != importlib.metadata.version("ray"):
                raise SystemExit(
                    "Gym servers and trainer must use the same Ray version; "
                    "derive the server runtime from the training lock before launch"
                )
            if runtime["python"][:2] != list(sys.version_info[:2]):
                raise SystemExit(
                    "Gym servers and trainer need matching Python minor versions"
                )
            checked.add(venv.resolve())
    if (
        not cfg["env"]["should_mask_flagged_samples"]
        or not gym["token_id_capture"]["enabled"]
        or gym["token_id_capture"].get("rebuild_response", True)
        or not cfg.get("token_capture", {}).get("enabled")
    ):
        raise SystemExit(
            "Training requires failure masking and native V2 token reconstruction"
        )
    sys.path.insert(0, str(root))
    from mmrl.worker_registry import register_memory_worker
    register_memory_worker(cfg)
    from nemo_rl.algorithms.single_controller_utils import MasterConfig
    from nemo_rl.algorithms.single_controller_utils.config import (
        validate_single_controller_config,
    )

    master = MasterConfig(**cfg)
    validate_single_controller_config(master)
    from nemo_rl.algorithms.single_controller_utils.setup import (
        validate_media_token_capture,
    )
    from nemo_rl.algorithms.utils import get_tokenizer

    processor = (
        get_tokenizer(master.policy["tokenizer"], get_processor=True)
        if master.policy.get("is_vlm")
        else None
    )
    validate_media_token_capture(master, processor)
    if args.check_only:
        print(
            "Native V2 configuration valid; no GPU update or remote acceptance was performed"
        )
    else:
        path = root / "examples/run_grpo_single_controller.py"
        spec = importlib.util.spec_from_file_location("mmrl_native_driver", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.argv = [str(path), "--config", str(Path(args.config).resolve())]
        module.main()


if __name__ == "__main__":
    main()
