import argparse
import json
from pathlib import Path

from .catalog import BENCHMARKS, LEGACY_BENCHMARKS
from .config import gym_config, training_config, validate_sibling_recipe
from .sandbox import provider_config
from .tasks import Task, write_rows
from .process_rewards import ProcessRewardConfig


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as output:
        json.dump(value, output, indent=2)
        output.write("\n")


def main():
    parser = argparse.ArgumentParser(prog="mmrl")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("benchmarks")
    sub.add_parser("check-access")
    p = sub.add_parser("import-legacy")
    p.add_argument("--root", required=True, help="Existing all_benchmark_rl directory")
    p.add_argument("--output", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--input", required=True, help="Canonical Task JSONL")
    p.add_argument("--output", required=True)
    p.add_argument("--mode", choices=["single", "multi", "mixed", "sibling-mixed"], required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epoch", type=int, default=0)
    p.add_argument("--max-output-tokens", type=int, default=32768)
    p.add_argument("--process-rewards", action="store_true",
                   help="Guide multi-turn agents to Gym's process-preview tool; pair with compose --process-rewards")
    p = sub.add_parser("compose")
    p.add_argument(
        "--base",
        required=True,
        help="Resolved V2 config JSON; keep its model and optimizer",
    )
    p.add_argument("--data", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--chartmimic-image", help="Digest-pinned Python 3.9 ChartMimic image")
    p.add_argument("--balanced-eleven", action="store_true",
                   help="Equal prompt exposure across all eleven benchmarks; single-turn data only")
    p.add_argument("--gateway", required=True)
    p.add_argument("--process-rewards", action="store_true",
                   help="Enable verified multi-turn process returns; single-turn rows keep final-only reward")
    p.add_argument("--harness-concurrency", type=int, default=8,
                   help="Concurrent trajectories per harness service")
    p.add_argument("--judge-concurrency", type=int, default=8,
                   help="Concurrent quality API requests per resource worker")
    p.add_argument("--sandbox-memory-mib", type=int, default=16384,
                   help="CPU sandbox memory per trajectory or verification session")
    p.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "benchmarks":
        for name, bench in BENCHMARKS.items():
            print(f"{name:24} multi={bench.multi_fraction:.0%}  {bench.rationale}")
    elif args.command == "check-access":
        provider_config()
        print("OpenSandbox configuration present; remote connectivity NOT tested")
    elif args.command == "prepare":
        with Path(args.input).open() as source:
            tasks = (Task(**json.loads(line)) for line in source if line.strip())
            write_rows(
                tasks,
                args.output,
                args.mode,
                seed=args.seed,
                epoch=args.epoch,
                max_output_tokens=args.max_output_tokens,
                process_rewards=args.process_rewards,
            )
    elif args.command == "import-legacy":
        from dataclasses import asdict

        with Path(args.output).open("x") as output:
            seen = set()
            for benchmark in LEGACY_BENCHMARKS:
                source = Path(args.root) / benchmark / "train_context_ready.jsonl"
                with source.open() as rows:
                    for line in rows:
                        row = json.loads(line)
                        task = Task.from_legacy_context(row["reward_model"])
                        identity = task.benchmark, task.task_id
                        if task.benchmark != benchmark or identity in seen:
                            raise ValueError(
                                "Unexpected benchmark or duplicate task identity"
                            )
                        seen.add(identity)
                        output.write(json.dumps(asdict(task)) + "\n")
    elif args.command == "compose":
        base = json.loads(Path(args.base).read_text())
        generation = base["policy"]["generation"]
        output_budget = generation["max_new_tokens"]
        context_window = generation["vllm_cfg"]["max_model_len"]
        sibling_mode = None
        benchmarks = set()
        with Path(args.data).open() as source:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                has_variants = "sibling_variants" in row
                if sibling_mode is not None and sibling_mode != has_variants:
                    raise ValueError("Do not mix sibling-expanded and group-routed dataset rows")
                sibling_mode = has_variants
                benchmarks.add(row.get("task", {}).get("benchmark"))
                if has_variants:
                    from .sibling_routes import apply_variant
                    validate_sibling_recipe(base)
                    apply_variant(dict(row, _rowidx=0), row["sibling_variants"])
                    if any(v["responses_create_params"]["max_output_tokens"] != output_budget
                           for v in row["sibling_variants"].values()):
                        raise ValueError("Sibling variant output budget differs from training")
                    if args.balanced_eleven:
                        raise ValueError("--balanced-eleven is single-turn only")
                if row.get("task", {}).get("benchmark") == "chartmimic" and not args.chartmimic_image:
                    raise ValueError("ChartMimic data requires --chartmimic-image; no renderer fallback is allowed")
                if (
                    row["responses_create_params"]["max_output_tokens"]
                    != output_budget
                ):
                    raise ValueError(
                        "Prepared data budget differs from the training recipe; re-prepare with --max-output-tokens"
                    )
        process_config = ProcessRewardConfig() if args.process_rewards else None
        services_config = gym_config(
            args.image,
            args.gateway,
            concurrency=args.harness_concurrency,
            judge_concurrency=args.judge_concurrency,
            sandbox_memory_mib=args.sandbox_memory_mib,
            max_output_tokens=output_budget,
            context_window=context_window,
            process_reward_config=process_config,
            chartmimic_image=args.chartmimic_image,
        )
        root = Path(args.output).resolve()
        services = root / "gym.json"
        trainer_config = training_config(
            base, services, args.data, root, process_reward_config=process_config,
            balanced_eleven=args.balanced_eleven,
        )
        if sibling_mode:
            train = trainer_config["data"]["train"]
            train["dataset_name"] = "mmrl.mixed_dataset.BalancedMixedGymDataset"
            configured = train.get("benchmarks")
            if configured is not None and set(configured) != benchmarks:
                raise ValueError("Configured benchmark selection differs from prepared mixed data")
            train["benchmarks"] = sorted(benchmarks)
        root.mkdir(parents=True, exist_ok=False)
        write_json(services, services_config)
        write_json(
            root / "train.json",
            trainer_config,
        )
        print(
            f"Composed {root / 'train.json'}; native validation and GPU acceptance still required"
        )


if __name__ == "__main__":
    main()
