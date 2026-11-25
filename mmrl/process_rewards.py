"""Bounded process objective and causal per-call rewards.

Inputs must come from trusted stage verification and selected capture records.
This calculator neither collects those records nor enables learner integration.
"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ProcessRewardConfig:
    progress_weight: float = 0.05
    progress_token_scale: int = 32768
    score_quantum: float = 0.025
    token_cost_per_1000: float = 0.0005
    render_cost: float = 0.001
    duplicate_render_cost: float = 0.002
    max_cost_fraction: float = 0.05
    execution_only_reward: float = 0.2

    def __post_init__(self):
        for name in ("progress_weight", "score_quantum", "token_cost_per_1000",
                     "render_cost", "duplicate_render_cost", "max_cost_fraction",
                     "execution_only_reward"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError("Process coefficients must be finite and nonnegative")
        if (type(self.progress_token_scale) is not int or self.progress_token_scale <= 0
                or not 0 < self.score_quantum <= 1 or self.progress_weight > 0.1
                or self.max_cost_fraction > 0.1 or self.execution_only_reward > 1):
            raise ValueError("Invalid process reward bounds")


@dataclass(frozen=True)
class StageScore:
    render_id: str
    call_index: int
    reward: float
    source_signature: str
    render_conditions_signature: str


@dataclass(frozen=True)
class ProcessSchedule:
    final_reward: float
    total_reward: float
    progress_bonus: float
    efficiency_cost: float
    raw_efficiency_cost: float
    duplicate_renders: int
    call_rewards: tuple[float, ...]
    returns_to_go: tuple[float, ...]
    terminal_settlement: float


def _score(value):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0 <= value <= 1):
        raise ValueError("Execution and quality reward must be finite in [0, 1]")
    return float(value)


def build_schedule(*, final_reward, assistant_tokens, stages=(), config=ProcessRewardConfig()):
    """Assign only prefix-dependent payments before a terminal settlement.

    The final-quality ceiling, failure gate and cost cap depend on the future.
    Settling them at the final call preserves the agreed bounded total without
    introducing future-dependent rewards into an earlier call's payment.
    """
    final = _score(final_reward)
    tokens = tuple(assistant_tokens)
    if not tokens or any(type(n) is not int or n <= 0 for n in tokens):
        raise ValueError("Each selected call requires a positive assistant-token count")
    stages = tuple(stages)
    render_ids, last_index = set(), -1
    for stage in stages:
        if (not isinstance(stage.render_id, str) or not stage.render_id
                or stage.render_id in render_ids or type(stage.call_index) is not int
                or not last_index <= stage.call_index < len(tokens)
                or stage.call_index < 0
                or not isinstance(stage.source_signature, str) or not stage.source_signature
                or not isinstance(stage.render_conditions_signature, str)
                or not stage.render_conditions_signature):
            raise ValueError("Stages require unique render IDs, ordered call indices and source/condition identities")
        _score(stage.reward)
        render_ids.add(stage.render_id)
        last_index = stage.call_index

    prefix_tokens, running_tokens = [], 0
    for count in tokens:
        running_tokens += count
        prefix_tokens.append(running_tokens)
    costs = [config.token_cost_per_1000 * count / 1000 for count in tokens]
    payments = [-cost for cost in costs]
    eligible = final > config.execution_only_reward + 1e-9
    quantum = config.score_quantum
    quantize = lambda value: math.floor((value + 1e-9) / quantum) * quantum
    ceiling = quantize(final)
    seen, duplicates = {}, 0
    capped_best, prefix_best, target_bonus = 0.0, 0.0, 0.0
    for stage in stages:
        identity = stage.source_signature, stage.render_conditions_signature
        duplicate = identity in seen
        duplicates += int(duplicate)
        # First score is authoritative for identical code and render conditions.
        score = seen.setdefault(identity, stage.reward)
        q = quantize(score)
        capped = min(ceiling, q)
        discount = math.exp(-prefix_tokens[stage.call_index] / config.progress_token_scale)
        if eligible:
            target_bonus += config.progress_weight * max(0.0, capped - capped_best) * discount
        capped_best = max(capped_best, capped)
        payments[stage.call_index] += config.progress_weight * max(0.0, q - prefix_best) * discount
        prefix_best = max(prefix_best, q)
        cost = config.render_cost + (config.duplicate_render_cost if duplicate else 0.0)
        costs[stage.call_index] += cost
        payments[stage.call_index] -= cost

    raw_cost = sum(costs)
    cost = min(config.max_cost_fraction * final, raw_cost) if eligible else 0.0
    total = final + target_bonus - cost
    settlement = total - sum(payments)
    payments[-1] += settlement
    returns, running = [0.0] * len(tokens), 0.0
    for index in range(len(tokens) - 1, -1, -1):
        running += payments[index]
        returns[index] = running
    if not math.isclose(running, total, abs_tol=1e-10):
        raise ValueError("Process reward accounting failed")
    return ProcessSchedule(final, total, target_bonus, cost, raw_cost, duplicates,
                           tuple(payments), tuple(returns), settlement)
