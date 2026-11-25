from copy import deepcopy
from types import SimpleNamespace
import unittest

from mmrl.process_capture import CapturedStageScore, bind_returns
from mmrl.process_rewards import ProcessRewardConfig


class ProcessCaptureTests(unittest.TestCase):
    def setUp(self):
        self.row = SimpleNamespace(
            rollout_id="episode", model_call_ids=["root", "terminal"],
            token_ids=[10, 11, 12, 20, 21, 22],
            token_mask=[0, 0, 1, 0, 1, 1], logprobs=[0, 0, -0.1, 0, -0.2, -0.3],
            prompt_len=2, link_spans=[("root", 2, 1), ("terminal", 1, 2)])

    def test_exact_assistant_spans_and_no_mutation(self):
        before = deepcopy(vars(self.row))
        result = bind_returns(self.row, final_reward=0.8, stages=(
            CapturedStageScore("episode", "root", "render1", 0.7, "source", "view"),))
        self.assertEqual(vars(self.row), before)
        self.assertEqual(result.assistant_tokens, (1, 2))
        self.assertEqual(result.token_returns, (0, 0, result.schedule.returns_to_go[0],
                                               0, result.schedule.returns_to_go[1], result.schedule.returns_to_go[1]))

    def test_final_only_reduces_to_existing_scalar_return(self):
        cfg = ProcessRewardConfig(progress_weight=0, token_cost_per_1000=0,
                                  render_cost=0, duplicate_render_cost=0)
        result = bind_returns(self.row, final_reward=0.8, config=cfg)
        self.assertEqual(result.token_returns, (0, 0, 0.8, 0, 0.8, 0.8))

    def test_foreign_or_discarded_stages_fail_closed(self):
        for rollout, call in (("other", "root"), ("episode", "retry"), ("episode", "auxiliary")):
            with self.assertRaises(ValueError):
                bind_returns(self.row, final_reward=0.8, stages=(
                    CapturedStageScore(rollout, call, "render", 0.6, "code", "view"),))

    def test_bad_boundary_or_call_identity_rejected(self):
        for changes in (
            {"link_spans": [("root", 1, 2), ("terminal", 1, 2)]},
            {"link_spans": [("root", 2, 1), ("terminal", 1, 1)]},
            {"model_call_ids": ["root", "root"]},
            {"prompt_len": 1},
            {"token_mask": [0, 0, 1, 1, 1, 1]},
        ):
            row = SimpleNamespace(**(vars(self.row) | changes))
            with self.assertRaises(ValueError):
                bind_returns(row, final_reward=0.8)


if __name__ == "__main__":
    unittest.main()
