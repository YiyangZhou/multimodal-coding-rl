# Multimodal Coding RL

Single-turn and agentic coding reinforcement learning built on [NeMo RL](https://github.com/NVIDIA-NeMo/RL) and [NeMo Gym](https://github.com/NVIDIA-NeMo/Gym).

## Rollout modes

| Mode | Behavior |
| --- | --- |
| `single` | Generate a direct answer without an agent harness. |
| `multi` | Run a supported agent harness with tool use. |
| `mixed` | Assign a direct or agentic route to each prompt group. |
| `sibling-mixed` | Generate eight answers per prompt: two direct, two OpenCode, two Codex, and two Hermes. |

The same-prompt mixture keeps prompt identities and route assignments consistent through retries. Training uses complete prompt groups. Generation capacity and optimizer batch size are configured independently.

## Components

- `mmrl/`: task formats, routing, sandbox execution, image and animation verification, rewards, trajectory capture, recovery helpers, and metrics.
- `integrations/nemo_gym/`: this project’s small resources-service adapter; no Gym framework source or data.
- `scripts/`: dataset preparation, configuration composition, launch validation, runtime binding, and patch installation.
- `patches/`: source-checked integration and performance patches.
- `containers/`: rendering and harness runtime definitions.
- `benchmarks/`: benchmark acquisition specifications.
- `tests/`: CPU contract tests and separate native/GPU acceptance checks.

Supported task adapters include Chart2Code, Img2SVG, Text2SVG, WebMMU Code Edit and Mockup2Code, Image2Struct, Design2Code, ChartMimic, Vision2Web, ManimBench, and 3DCodeBench. Route availability is defined in `mmrl/catalog.py`; dataset support does not imply every task supports every harness.

Gym is an external dependency. This repository does not distribute a Gym checkout, Gym patches, datasets, or deployment configurations. Runtime checks require externally supplied verification manifests through `MMRL_EXTERNAL_GYM_PATCH_DIR`; a stock upstream installation has not yet been validated with all integration features.

## Configuration

Use your own checkpoint, tokenizer, dataset, output directory, and service credentials. Copy `configs/examples/team-runtime.env.example` to a private location and fill in the values. Credentials are kept outside agent sandboxes.

Quality scoring uses an OpenAI-compatible Responses API with image input and structured output. Set `OPENAI_BASE_URL`, `MMRL_JUDGE_MODEL`, and either `OPENAI_API_KEY` or the comma-separated `OPENAI_API_KEYS`. Check that your provider supports these API features.

## Preparing data

Activate the matching NeMo RL/Gym environment, then run:

```bash
source scripts/env.sh
python -m mmrl.cli benchmarks
python -m mmrl.cli prepare \
  --input tasks.jsonl --output runs/train.jsonl \
  --mode sibling-mixed --max-output-tokens 32768
```

Use `single`, `multi`, or `mixed` to select another mode. Input rows use the canonical `Task` fields in `mmrl/tasks.py`. Referenced assets must remain readable by the rendering services.

Compose the services and training configuration from a resolved NeMo RL recipe:

```bash
python -m mmrl.cli compose \
  --base resolved-train.json --data runs/train.jsonl \
  --image "$MMRL_SANDBOX_IMAGE" \
  --gateway "$MMRL_CAPTURE_GATEWAY" \
  --output runs/config
```

For `sibling-mixed`, the recipe must use eight generations per prompt. Preserve the intended benchmark weights and sampling seed. Do not use `--balanced-eleven` for a same-prompt mixture: that option builds a separate single-turn dataset.

## Training

The package integrates with an installed NeMo RL runtime; it does not replace its PyTorch, vLLM, or distributed training dependencies. The patch manifests require exact source versions and intentionally reject incompatible checkouts. The native patches are snapshot-specific. Their prerequisites do not currently match the public upstream revisions checked during export, so a fresh checkout of upstream `main` is not a supported installation yet. Data preparation and CPU contracts are tested separately; GPU training against a public-only dependency stack remains unverified. See [runtime compatibility](docs/runtime-compatibility.md).

After installing a compatible runtime and binding the Gym environment, validate and launch inside a GPU allocation:

```bash
python scripts/train.py --config runs/config/train.json \
  --rl-root "$NEMO_RL_ROOT" --check-only
python scripts/train.py --config runs/config/train.json \
  --rl-root "$NEMO_RL_ROOT"
```

`--check-only` validates configuration and runtime contracts; it does not prove GPU training correctness. Native checks in `tests/check_*` cover capture, masking, recovery, and learner integration with the required runtime installed.

## Reward

Executable outputs receive `0.2 + 0.8 × mean(task_fidelity, visual_fidelity)`. Model execution failures receive zero. Infrastructure and judge failures are excluded and reported separately. Truncation validity and the optional length penalty are separate controls. Ground-truth source and judge-only reference assets must not enter policy inputs.

## Tests

Install CPU test dependencies in a separate environment:

```bash
python -m pip install -e '.[test]'
python -m pytest tests -q
```

GPU and remote sandbox checks require their respective environments and credentials. CPU tests do not substitute for those checks.
