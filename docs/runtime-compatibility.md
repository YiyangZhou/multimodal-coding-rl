# Runtime compatibility

The repository includes task adapters, scoring, trajectory capture, single-turn and agentic route selection, same-prompt mixed rollouts, launch scripts, and native integration patches. Gym framework code, Gym patches, datasets, credentials, and deployment records are excluded.

## What is verified

CPU contract tests cover routing, reward handling, failure masks, sibling identities, and configuration validation. The data-preparation CLI has been exercised in all four rollout modes. Same-prompt mixed preparation assigns two siblings each to direct generation, OpenCode, Codex, and Hermes.

## What requires a compatible external runtime

Training uses NeMo RL's single-controller runtime and a separately installed Gym service. Native patch manifests check exact source hashes and reject incompatible or partially patched trees. The historical prerequisites have not been reproduced from the public upstream revisions inspected so far. Do not bypass these checks or assume that CPU tests validate GPU training.

Gym-specific verification manifests must be supplied through `MMRL_EXTERNAL_GYM_PATCH_DIR`. Image construction additionally requires `--gym-source-lock`, a JSON mapping of relative source paths to SHA256 hashes. Neither setting provides the missing implementation; obtain a compatible external runtime before attempting a launch.

The four rollout modes share the same runtime prerequisites. Source preparation works without launching Gym; generation, rendering, and learner updates require the corresponding services and model dependencies.

## Local checks

```bash
python -m pytest tests -q
python scripts/check_runtime_sources.py --rl-root "$NEMO_RL_ROOT"
```

The second command reports source compatibility, not training correctness. Remote sandbox checks and native GPU acceptance must pass separately before using a new dependency stack for training.
