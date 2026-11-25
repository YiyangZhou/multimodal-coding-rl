# Runtime compatibility

The repository includes task adapters, scoring, trajectory capture, single-turn and agentic route selection, same-prompt mixed rollouts, launch scripts, and native integration patches. Gym framework code, Gym patches, datasets, credentials, and deployment records are excluded.

## What is verified

CPU contract tests cover routing, reward handling, failure masks, sibling identities, and configuration validation. The data-preparation CLI has been exercised in all four rollout modes. Same-prompt mixed preparation assigns two siblings each to direct generation, OpenCode, Codex, and Hermes.

## What requires a compatible external runtime

Training uses NeMo RL's single-controller runtime and a separately installed Gym service. Native patch manifests check exact source hashes and reject incompatible or partially patched trees. The historical patch chain remains snapshot-specific. An experimental composite port is available for the public revisions listed below; it is a separate installation profile. Do not bypass these checks or assume that CPU tests validate GPU training.

Gym-specific verification manifests must be supplied through `MMRL_EXTERNAL_GYM_PATCH_DIR`. Image construction additionally requires `--gym-source-lock`, a JSON mapping of relative source paths to SHA256 hashes. Neither setting provides the missing implementation; obtain a compatible external runtime before attempting a launch.

The four rollout modes share the same runtime prerequisites. Source preparation works without launching Gym; generation, rendering, and learner updates require the corresponding services and model dependencies.

## Local checks

```bash
python -m pytest tests -q
python scripts/check_runtime_sources.py --rl-root "$NEMO_RL_ROOT"
```

The second command reports source compatibility, not training correctness. Remote sandbox checks and native GPU acceptance must pass separately before using a new dependency stack for training.

## Experimental public-source profile

The composite patch includes captured media, continuous generation, multi-turn fixes, same-prompt sibling routing, and the original-history bridge. It is checked against:

| Dependency | Commit |
| --- | --- |
| NeMo RL | `31b4c0295cff085e74f2724e2cf057e5b3e3ae04` |
| AutoModel | `72daceffa54d77783e1f73a5ac165eb39b6b5ea1` |

Start with a separate clean checkout:

```bash
git clone https://github.com/NVIDIA-NeMo/RL.git nemo-rl-public
cd nemo-rl-public
git checkout 31b4c0295cff085e74f2724e2cf057e5b3e3ae04
git submodule update --init 3rdparty/Automodel-workspace/Automodel
export NEMO_RL_ROOT="$PWD"
cd ../multimodal-coding-rl
python scripts/apply_capture_patch.py --rl-root "$NEMO_RL_ROOT" --public-runtime
python scripts/apply_capture_patch.py --rl-root "$NEMO_RL_ROOT" --public-runtime --check
```

Do not apply the historical native layers on top of this composite profile. The installer checks both Git revisions and the affected file hashes, stages and verifies the patch before installation, and accepts repeat invocations only when all expected files match.

Add `--public-runtime` to `scripts/train.py` when using this checkout. This selects the composite source check for all rollout modes; external Gym checks still apply. Install the model dependencies and provide the compatible external Gym service before attempting launch.

Clean installation, repeat installation, source verification, and rejection of modified sources have been tested. GPU training on this dependency stack remains unverified; this profile is experimental.

### Project harness adapter

The optional `--project-harness-adapter` flag selects `responses_api_agents/mm_harness`, a project-owned subclass of the external Gym harness agent. Use it consistently with `mmrl compose`, `scripts/bind_runtime.py`, and `scripts/train.py`. Keep the repository's `integrations/nemo_gym` directory on `NEMO_GYM_EXTRA_ROOTS` so Gym can discover the component.

This adapter turns only an explicit upstream runner failure into a failed response. The coding resources server then returns a masked sample. Other exceptions keep their original behavior. It contains no Gym framework implementation.

CPU response-model and resource-verifier checks passed against Gym commit `17abf2ddc76f74e40d0ba5a15b0401710a46db26`, the submodule pinned by the NeMo RL revision above, using Python 3.13.14. This is not full Gym or GPU acceptance. Trainer and server environments must still have matching Python minor and Ray versions; original-history capture still requires separately verified external support.

```bash
python tests/check_harness_failure_gym.py
```

Run this check in the external Gym environment with this project installed. It tests failed responses and masks without launching a sandbox or requesting model inference.
