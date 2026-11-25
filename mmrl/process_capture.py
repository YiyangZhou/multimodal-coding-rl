"""Bind process returns to Gym's verified terminal-chain assistant spans.

Call only after Gym verify_and_linearize, then retain the native media, policy
version and learner eligibility checks. This module does not enable training.
"""
from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

from .process_rewards import ProcessRewardConfig, ProcessSchedule, StageScore, build_schedule

if TYPE_CHECKING:
    from nemo_gym.token_id_capture.staging.rebuild import LinearizedRow


@dataclass(frozen=True)
class CapturedStageScore:
    rollout_id: str
    model_call_id: str
    render_id: str
    reward: float
    source_signature: str
    render_conditions_signature: str


@dataclass(frozen=True)
class BoundProcessReturns:
    rollout_id: str
    model_call_ids: tuple[str, ...]
    assistant_tokens: tuple[int, ...]
    token_returns: tuple[float, ...]
    schedule: ProcessSchedule


def bind_verifier_payload(row: "LinearizedRow", *, final_reward, payload):
    """Decode trusted verification metadata after native chain validation.

    The caller supplies the independent final reward and verified native row.
    Payload identities and scores must agree; metadata cannot replace either.
    This function is a learner-boundary adapter, not a receipt transport hook.
    """
    expected = {"version", "rollout_id", "final_reward", "stages", "config"}
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("Incomplete or unknown process verification envelope")
    if type(payload["version"]) is not int or payload["version"] != 1:
        raise ValueError("Unsupported process verification version")
    if payload["rollout_id"] != row.rollout_id:
        raise ValueError("Process evidence belongs to a different rollout")
    for score in (final_reward, payload["final_reward"]):
        if (isinstance(score, bool) or not isinstance(score, (int, float))
                or not math.isfinite(score) or not 0 <= score <= 1):
            raise ValueError("Independent final reward must be finite in [0, 1]")
    if final_reward != payload["final_reward"]:
        raise ValueError("Process evidence disagrees with the independent final reward")
    if not isinstance(payload["stages"], list) or not isinstance(payload["config"], dict):
        raise ValueError("Expected explicit stage records and reward coefficients")
    stage_keys = set(CapturedStageScore.__dataclass_fields__)
    if any(not isinstance(stage, dict) or set(stage) != stage_keys for stage in payload["stages"]):
        raise ValueError("Malformed process stage record")
    if set(payload["config"]) != set(ProcessRewardConfig.__dataclass_fields__):
        raise ValueError("Process verification must carry the complete reward configuration")
    return bind_returns(row, final_reward=final_reward,
                        stages=tuple(CapturedStageScore(**stage) for stage in payload["stages"]),
                        config=ProcessRewardConfig(**payload["config"]))


def bind_returns(row: "LinearizedRow", *, final_reward, stages=(), config=ProcessRewardConfig()):
    """Use native selected call IDs, never tool count or response text offsets.

    Stage scores are trusted verifier inputs. A score attached to an auxiliary
    or discarded retry branch is rejected instead of attributed to another call.
    Returned values are rewards-to-go, not already-normalized GRPO advantages.
    """
    call_ids = tuple(row.model_call_ids)
    if (not row.rollout_id or not call_ids or len(set(call_ids)) != len(call_ids)
            or len(row.link_spans) != len(call_ids)
            or len(row.token_ids) != len(row.token_mask)
            or len(row.token_ids) != len(row.logprobs)):
        raise ValueError("Incomplete verified rollout layout")
    offset, spans, counts = 0, [], []
    for call_id, (span_id, carry, generated) in zip(call_ids, row.link_spans, strict=True):
        if (span_id != call_id or type(carry) is not int or carry < 0
                or type(generated) is not int or generated <= 0):
            raise ValueError("Invalid native selected-call span")
        start, end = offset + carry, offset + carry + generated
        if (end > len(row.token_mask) or any(m != 0 for m in row.token_mask[offset:start])
                or any(m != 1 for m in row.token_mask[start:end])):
            raise ValueError("Call spans disagree with the captured assistant mask")
        spans.append((start, end))
        counts.append(generated)
        offset = end
    if offset != len(row.token_ids) or row.prompt_len != row.link_spans[0][1]:
        raise ValueError("Native spans do not cover the exact canonical sequence")
    positions = {call_id: index for index, call_id in enumerate(call_ids)}
    bound = []
    for stage in stages:
        if stage.rollout_id != row.rollout_id or stage.model_call_id not in positions:
            raise ValueError("Stage does not belong to this selected terminal chain")
        bound.append(StageScore(stage.render_id, positions[stage.model_call_id], stage.reward,
                                stage.source_signature, stage.render_conditions_signature))
    schedule = build_schedule(final_reward=final_reward, assistant_tokens=counts,
                              stages=bound, config=config)
    token_returns = [0.0] * len(row.token_ids)
    for (start, end), reward in zip(spans, schedule.returns_to_go, strict=True):
        token_returns[start:end] = [reward] * (end - start)
    return BoundProcessReturns(row.rollout_id, call_ids, tuple(counts),
                               tuple(token_returns), schedule)


def bind_native_returns(row: "LinearizedRow", *, final_reward, payload):
    """Bridge verified Gym spans to the optional native learner return contract."""
    from nemo_rl.experience.process_returns import ProcessReturns

    bound = bind_verifier_payload(row, final_reward=final_reward, payload=payload)
    return ProcessReturns(bound.schedule.total_reward, bound.token_returns)
