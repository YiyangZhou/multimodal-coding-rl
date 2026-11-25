"""Equal benchmark exposure through row indices, without copying image payloads."""

import json
import random

LEGACY_BENCHMARKS = ('chart2code', 'mmsvg_img2svg', 'webmmu_code_edit', 'webmmu_mockup2code', 'image2struct_latex', 'mmsvg_text2svg', 'design2code_full')


ELEVEN_BENCHMARKS = LEGACY_BENCHMARKS + (
    "3dcodebench", "chartmimic", "vision2web", "manimbench_v1",
)


def balanced_indices(raw_rows, *, seed=42, batch_size=8, benchmark_weights=None, benchmarks=None, excluded_tasks=None):
    """Build one equal-exposure pass; the native loader shuffles and checkpoints it.

    Each source row appears before its benchmark cycles. Small benchmarks are
    oversampled to the largest one's size, rounded up so drop_last loses no rows.
    This balances dispatched prompts over a pass, not successful completions or
    every individual optimizer batch in an asynchronous ready-first run.
    """
    if type(seed) is not int or type(batch_size) is not int or batch_size < 1:
        raise ValueError("seed must be an integer and batch_size a positive integer")
    selected = ELEVEN_BENCHMARKS if benchmarks is None else tuple(benchmarks)
    if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(ELEVEN_BENCHMARKS):
        raise ValueError("Explicit benchmark selection must be nonempty, unique and known")
    exclusions = {}
    for item in excluded_tasks or []:
        if not isinstance(item, dict) or set(item) != {"benchmark", "task_id", "reason"}:
            raise ValueError("Excluded tasks require benchmark, task_id and reason")
        identity = item["benchmark"], item["task_id"]
        if (identity[0] not in selected or not isinstance(identity[1], str)
                or not identity[1].strip() or not isinstance(item["reason"], str)
                or not item["reason"].strip() or identity in exclusions):
            raise ValueError("Invalid or duplicate excluded task")
        exclusions[identity] = dict(item)
    excluded_seen = set()
    groups = {name: [] for name in selected}
    seen = set()
    for index, raw in enumerate(raw_rows):
        row = json.loads(raw)
        task = row.get("task", {})
        name, task_id = task.get("benchmark"), task.get("task_id")
        if name not in groups:
            raise ValueError(f"Unknown benchmark at row {index}")
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError(f"Missing task identity at row {index}")
        if (row.get("route"), row.get("agent_ref", {}).get("name")) not in {("single", "mm_single"), ("opencode", "mm_opencode"), ("codex", "mm_codex"), ("hermes", "mm_hermes")}:
            raise ValueError(f"Expected a supported single or multi-turn route at row {index}")
        variants = row.get("sibling_variants")
        routes = {"single", "opencode", "codex", "hermes"}
        if not isinstance(variants, dict) or set(variants) != routes:
            raise ValueError(f"Missing same-prompt route variants at row {index}")
        for route, variant in variants.items():
            if (variant.get("task") != task or variant.get("route") != route
                    or variant.get("agent_ref") != {"type": "responses_api_agents", "name": "mm_" + route}
                    or not variant.get("responses_create_params", {}).get("input")):
                raise ValueError(f"Invalid same-prompt variant at row {index}: {route}")
        identity = name, task_id
        if identity in seen:
            raise ValueError(f"Duplicate task identity at row {index}")
        seen.add(identity)
        if identity in exclusions:
            excluded_seen.add(identity)
            continue
        groups[name].append(index)
    if excluded_seen != set(exclusions):
        raise ValueError("Configured exclusion was not found in source rows")
    missing = [name for name, rows in groups.items() if not rows]
    if missing:
        raise ValueError(f"Missing benchmarks: {', '.join(missing)}")
    quota = ((max(map(len, groups.values())) + batch_size - 1) // batch_size) * batch_size
    weights = benchmark_weights
    if weights is not None:
        if set(weights) != set(groups) or any(type(v) is not int or v <= 0 for v in weights.values()):
            raise ValueError("benchmark_weights must contain positive integer weights for all selected tasks")
        units = max((len(groups[name]) + weight - 1) // weight for name, weight in weights.items())
        units = ((units + batch_size - 1) // batch_size) * batch_size
        quotas = {name: units * weights[name] for name in groups}
    else:
        quotas = {name: quota for name in groups}
    rng = random.Random(seed)
    indices = []
    for name, rows in groups.items():
        remaining = quotas[name]
        while remaining:
            cycle = rows.copy()
            rng.shuffle(cycle)
            take = min(remaining, len(cycle))
            indices.extend(cycle[:take])
            remaining -= take
    report = {
        "type": "equal_per_benchmark",
        "benchmarks": list(selected),
        "source_counts": {name: len(rows) for name, rows in groups.items()},
        "prompts_per_benchmark_per_pass": quota,
        "prompts_per_pass": len(indices),
        "weight_each": 1 / len(groups),
        "seed": seed,
        "batch_size": batch_size,
    }
    if weights is not None:
        report.update(type="weighted_per_benchmark", prompts_per_benchmark_per_pass=quotas,
                      weight_each=None, benchmark_weights=dict(weights),
                      proportions={name: value / sum(weights.values()) for name, value in weights.items()})
    if exclusions:
        report["excluded_tasks"] = list(exclusions.values())
    return indices, report
