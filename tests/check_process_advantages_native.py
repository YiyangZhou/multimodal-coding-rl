"""CPU checks against the locked native GRPO estimator; no optimizer update."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from mmrl.process_advantages import compute_process_advantages
from nemo_rl.algorithms.advantage_estimator import GRPOAdvantageEstimator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prompts = torch.tensor([[0], [0], [0], [1], [1]])
    rewards = torch.tensor([0.3, 0.6, 0.9, 0.7, 0.7])
    mask = torch.tensor([[0, 1, 0, 1, 1]] * 5, dtype=torch.float32)
    cases = 0
    for loo in (False, True):
        for normalize in (False, True):
            cfg = SimpleNamespace(name="grpo", use_leave_one_out_baseline=loo,
                                  normalize_rewards=normalize)
            native = GRPOAdvantageEstimator(cfg, None)
            for valid in (torch.ones(5), torch.tensor([1., 1., 0., 1., 0.]), torch.zeros(5)):
                reference = native.compute_advantage(prompts, rewards, mask, valid_mask=valid)
                reference = reference * mask * valid.unsqueeze(1)
                result = compute_process_advantages(prompt_ids=prompts, rewards=rewards,
                    token_returns=rewards.unsqueeze(1).expand_as(mask) * mask,
                    mask=mask, valid_mask=valid, estimator_config=cfg)
                torch.testing.assert_close(result, reference, rtol=0, atol=0)
                cases += 1
    cfg = SimpleNamespace(name="grpo", use_leave_one_out_baseline=True, normalize_rewards=False)
    process_returns = rewards.unsqueeze(1).expand_as(mask).clone() * mask
    process_returns[0, 3:] -= 0.05
    result = compute_process_advantages(prompt_ids=prompts, rewards=rewards,
        token_returns=process_returns, mask=mask, valid_mask=torch.ones(5), estimator_config=cfg)
    reference = GRPOAdvantageEstimator(cfg, None).compute_advantage(
        prompts, rewards, mask, valid_mask=torch.ones(5)) * mask
    torch.testing.assert_close(result[1:], reference[1:])
    torch.testing.assert_close(result[0, :3], reference[0, :3])
    torch.testing.assert_close(result[0, 3:], reference[0, 3:] - 0.05)
    # An excluded failed episode and a singleton group remain fully inactive.
    rewards[2] = float('nan')
    process_returns[2] = float('nan')
    result = compute_process_advantages(prompt_ids=prompts, rewards=rewards,
        token_returns=process_returns, mask=mask,
        valid_mask=torch.tensor([1., 1., 0., 1., 0.]), estimator_config=cfg)
    assert torch.isfinite(result).all() and not result[2:].any()
    assert not torch.cuda.is_initialized()
    import inspect
    from nemo_rl.algorithms.utils import calculate_baseline_and_std_per_prompt
    files = [Path(__file__), Path(inspect.getfile(compute_process_advantages)),
             Path(inspect.getfile(GRPOAdvantageEstimator)),
             Path(inspect.getfile(calculate_baseline_and_std_per_prompt))]
    proof = {"passed": True, "native_final_only_equivalence_cases": cases,
             "late_call_only_credit_changed": True, "excluded_and_singleton_rows_zero": True,
             "cuda_initialized": False, "learner_pipeline_enabled": False,
             "scope": "Native CPU baseline/normalization equivalence and per-call credit arithmetic, not an actual policy update.",
             "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    with args.output.open("x") as stream:
        json.dump(proof, stream, indent=2)
        stream.write("\n")
    print(json.dumps({k: v for k, v in proof.items() if k != "source_sha256"}))


if __name__ == "__main__":
    main()
