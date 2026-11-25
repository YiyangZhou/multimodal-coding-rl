"""Bind prebuilt server environments without resolving or installing packages."""

import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--venv", required=True, help="Existing locked Gym environment")
p.add_argument("--project-harness-adapter", action="store_true", help="Use the mm_harness component for agent rollouts")
p.add_argument("--output", required=True)
args = p.parse_args()
venv = Path(args.venv).resolve()
if not (venv / "bin/python").is_file() or not (venv / "bin/activate").is_file():
    p.error("Expected a prebuilt environment with python and activate")
root = Path(args.output).resolve()
root.mkdir(parents=True, exist_ok=False)
for component in (
    "resources_servers/mm_coding",
    "responses_api_agents/simple_agent",
    ("responses_api_agents/mm_harness" if args.project_harness_adapter
         else "responses_api_agents/harness_agent"),
    "responses_api_models/vllm_model",
):
    target = root / component
    target.mkdir(parents=True)
    (target / ".venv").symlink_to(venv, target_is_directory=True)
print(json.dumps({"uv_venv_dir": str(root), "skip_venv_if_present": True}, indent=2))
