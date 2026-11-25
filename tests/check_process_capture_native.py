"""Exercise the binding after the pinned Gym's real receipt verifier.

Uses Gym's synthetic branched-capture fixture; no model or learner is run.
"""
import argparse
import copy
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gym-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(args.gym_root)]
    from mmrl.process_capture import CapturedStageScore, bind_returns, bind_verifier_payload
    from mmrl.process_rewards import ProcessRewardConfig
    from nemo_gym.token_id_capture.staging.rebuild import verify_and_linearize

    fixture_path = args.gym_root / "tests/unit_tests/test_token_capture_staging_rebuild.py"
    spec = importlib.util.spec_from_file_location("native_capture_fixture", fixture_path)
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    receipt, snapshots = fixture._branched()
    row = verify_and_linearize(receipt, snapshots)
    assert row.model_call_ids == ["root", "main"]
    assert {r.model_call_id for r in receipt.manifest} == {"root", "main", "sibling"}
    stage = CapturedStageScore(row.rollout_id, "root", "render-root", 0.5, "code", "view")
    result = bind_returns(row, final_reward=0.8, stages=(stage,))
    assert result.assistant_tokens == (1, 1)
    assert result.token_returns == (0, 0, result.schedule.returns_to_go[0], 0,
                                     result.schedule.returns_to_go[1])
    assert result.model_call_ids == ("root", "main")
    payload = {"version": 1, "rollout_id": row.rollout_id, "final_reward": .8,
               "stages": [asdict(stage)], "config": asdict(ProcessRewardConfig())}
    assert bind_verifier_payload(row, final_reward=.8, payload=payload) == result
    mutations = [
        {"version": True}, {"version": 2}, {"rollout_id": "another-rollout"},
        {"final_reward": .9}, {"final_reward": float("nan")},
        {"config": {}}, {"stages": [asdict(stage) | {"model_call_id": "sibling"}]},
        {"stages": [asdict(stage) | {"reward": float("nan")}]},
        {"stages": [asdict(stage) | {"untrusted_extra": 1}]}, {"untrusted_extra": 1},
    ]
    for mutation in mutations:
        candidate = copy.deepcopy(payload) | mutation
        try:
            bind_verifier_payload(row, final_reward=.8, payload=candidate)
        except ValueError:
            pass
        else:
            raise AssertionError("Malformed process evidence was accepted: " + repr(mutation))
    try:
        bind_returns(row, final_reward=0.8, stages=(CapturedStageScore(
            row.rollout_id, "sibling", "wrong-branch", 0.9, "code", "view"),))
    except ValueError:
        pass
    else:
        raise AssertionError("Off-chain reward was accepted")
    sources = [fixture_path, args.gym_root / "nemo_gym/token_id_capture/staging/rebuild.py",
               Path(__file__).resolve().parents[1] / "mmrl/process_capture.py",
               Path(__file__).resolve().parents[1] / "mmrl/process_rewards.py"]
    report = {"passed": True, "synthetic": True, "model_inference": False,
              "selected_calls": list(result.model_call_ids), "assistant_tokens": list(result.assistant_tokens),
              "off_chain_score_rejected": True, "tool_tokens_have_zero_return": True,
              "verifier_payload_roundtrip": True, "invalid_payload_cases_rejected": len(mutations),
              "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
              "scope": "Native Gym receipt verification and terminal-chain return binding only; no media verification, stage collection, GRPO normalization or learner consumption."}
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({k: v for k, v in report.items() if k != "source_sha256"}))


if __name__ == "__main__":
    main()
