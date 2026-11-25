"""Verify actual Gym imports, including component search-path precedence."""

import hashlib
import importlib
import json
import os
from pathlib import Path


def main():
    directory = os.environ.get("MMRL_EXTERNAL_GYM_PATCH_DIR", "")
    manifest_path = Path(directory) / "gym-original-history.json"
    if not directory or not manifest_path.is_file():
        raise SystemExit("Set MMRL_EXTERNAL_GYM_PATCH_DIR to external Gym verification manifests; Gym patches are not distributed here.")
    manifest = json.loads(manifest_path.read_text())
    # Gym adjusts sys.path during package initialization; inspect imports after
    # that adjustment, not just files in the requested checkout.
    importlib.import_module("nemo_gym.base_responses_api_model")
    origins = {}
    for name, spec in manifest["files"].items():
        module = importlib.import_module(name.removesuffix(".py").replace("/", "."))
        source = Path(module.__file__).resolve()
        if hashlib.sha256(source.read_bytes()).hexdigest() != spec["after_sha256"]:
            raise SystemExit("Original-history import mismatch: " + str(source) +
                             "; check PYTHONPATH and NEMO_GYM_EXTRA_ROOTS")
        origins[name] = str(source)
    print(json.dumps({"checked_imports": len(origins), "origins": origins}, sort_keys=True))


if __name__ == "__main__":
    main()
