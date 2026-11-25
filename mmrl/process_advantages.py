"""Per-call returns with NeMo RL's original episode-level GRPO baseline.

The optional native-process-returns patch invokes this hook after admission masks.
"""


def compute_process_advantages(*, prompt_ids, rewards, token_returns, mask,
                               valid_mask, estimator_config):
    import torch
    from nemo_rl.algorithms.utils import calculate_baseline_and_std_per_prompt

    if estimator_config.name != "grpo":
        raise ValueError("Process return integration currently covers GRPO only")
    if (rewards.ndim != 1 or token_returns.ndim != 2
            or mask.shape != token_returns.shape or token_returns.shape[0] != len(rewards)
            or valid_mask.shape != rewards.shape or prompt_ids.ndim != 2
            or prompt_ids.shape[0] != len(rewards)
            or not rewards.is_floating_point() or not token_returns.is_floating_point()
            or rewards.device != token_returns.device or mask.device != rewards.device
            or valid_mask.device != rewards.device):
        raise ValueError("Expected aligned episode rewards and canonical token-return rows")
    if not bool((((valid_mask == 0) | (valid_mask == 1)).all()).item()):
        raise ValueError("Episode validity must be binary")
    if not bool((((mask == 0) | (mask == 1)).all()).item()):
        raise ValueError("Assistant token mask must be binary")
    valid = valid_mask.bool()
    selected = valid.unsqueeze(1) & mask.bool()
    if (not bool(torch.isfinite(rewards[valid]).all().item())
            or not bool(torch.isfinite(token_returns[selected]).all().item())):
        raise ValueError("Valid trajectories require finite rewards and returns")
    # Excluded infrastructure/capture failures cannot contaminate sibling stats.
    totals = torch.where(valid, rewards, torch.zeros_like(rewards))
    returns = torch.where(selected, token_returns, torch.zeros_like(token_returns))
    baseline, std, trivial = calculate_baseline_and_std_per_prompt(
        prompt_ids, totals, valid.float(),
        leave_one_out_baseline=estimator_config.use_leave_one_out_baseline)
    advantages = returns - baseline.unsqueeze(1)
    if estimator_config.normalize_rewards:
        normalize = (std > 0) & ~trivial
        scale = torch.where(normalize, std + 1e-6, torch.ones_like(std))
        advantages = advantages / scale.unsqueeze(1)
    # Preserve native suppression of groups with fewer than two eligible
    # episodes, even if an individual episode has nonconstant process returns.
    eligible_group = torch.zeros_like(valid)
    for prompt in torch.unique(prompt_ids, dim=0):
        members = (prompt_ids == prompt).all(1).to(valid.device)
        if int(valid[members].sum().item()) > 1:
            eligible_group[members] = True
    return torch.where(selected & eligible_group.unsqueeze(1), advantages,
                       torch.zeros_like(advantages))
