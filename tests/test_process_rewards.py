import math
import unittest

from mmrl.process_rewards import ProcessRewardConfig, StageScore, build_schedule


class ProcessRewardTests(unittest.TestCase):
    def test_agreed_bounded_objective_and_failure_gate(self):
        stages = (StageScore("a", 0, 0.5, "source-a", "view"),
                  StageScore("b", 1, 0.9, "source-b", "view"))
        result = build_schedule(final_reward=0.8, assistant_tokens=(1000, 2000), stages=stages)
        bonus = 0.05 * (0.5 * math.exp(-1000 / 32768) + 0.3 * math.exp(-3000 / 32768))
        self.assertAlmostEqual(result.progress_bonus, bonus)
        self.assertAlmostEqual(result.efficiency_cost, 0.0035)
        self.assertAlmostEqual(result.total_reward, 0.8 + bonus - 0.0035)
        for final in (0, 0.2):
            failed = build_schedule(final_reward=final, assistant_tokens=(1000, 2000), stages=stages)
            self.assertEqual(failed.progress_bonus, 0)
            self.assertEqual(failed.efficiency_cost, 0)
            self.assertAlmostEqual(sum(failed.call_rewards), final)

    def test_duplicate_source_reuses_first_score(self):
        result = build_schedule(final_reward=0.8, assistant_tokens=(1, 1), stages=(
            StageScore("a", 0, 0.5, "same-code", "same-view"),
            StageScore("b", 1, 1.0, "same-code", "same-view")))
        self.assertEqual(result.duplicate_renders, 1)
        self.assertAlmostEqual(result.progress_bonus, 0.025 * math.exp(-1 / 32768))
        self.assertAlmostEqual(result.raw_efficiency_cost, 0.004001)

    def test_final_failure_cannot_change_prefix_payment(self):
        args = dict(assistant_tokens=(1000, 1000), stages=(StageScore("a", 0, 0.9, "code", "view"),))
        failure = build_schedule(final_reward=0, **args)
        success = build_schedule(final_reward=0.8, **args)
        self.assertEqual(failure.call_rewards[0], success.call_rewards[0])
        self.assertNotEqual(failure.call_rewards[1], success.call_rewards[1])

    def test_causal_returns_match_exact_policy_gradient(self):
        p1, p2 = 0.37, 0.61
        exact, causal = [0.0, 0.0], [0.0, 0.0]
        for first in (0, 1):
            for second in (0, 1):
                probability = (p1 if first else 1 - p1) * (p2 if second else 1 - p2)
                final = (0.9 if first else 0.6) if second else (0.3 if first else 0.0)
                result = build_schedule(final_reward=final,
                    assistant_tokens=(100 + 100 * first, 150 + 200 * second),
                    stages=(StageScore("a", 0, 0.9 if first else 0.4, "code", "view"),))
                for index, score in enumerate((first - p1, second - p2)):
                    exact[index] += probability * score * result.total_reward
                    causal[index] += probability * score * result.returns_to_go[index]
        for a, b in zip(exact, causal):
            self.assertAlmostEqual(a, b, places=12)

    def test_invalid_capture_or_scores_rejected(self):
        for counts in ((), (True,), (0,), (-1,), (1.5,)):
            with self.assertRaises(ValueError):
                build_schedule(final_reward=0.8, assistant_tokens=counts)
        for final in (True, math.nan, math.inf, -1, 1.1):
            with self.assertRaises(ValueError):
                build_schedule(final_reward=final, assistant_tokens=(1,))
        for stages in (
            (StageScore("a", -1, 0.5, "code", "view"),),
            (StageScore("a", 1, 0.5, "code", "view"),),
            (StageScore("a", 0, math.nan, "code", "view"),),
            (StageScore("a", 0, 0.5, "code", "view"),) * 2,
        ):
            with self.assertRaises(ValueError):
                build_schedule(final_reward=0.8, assistant_tokens=(1,), stages=stages)
        with self.assertRaises(ValueError):
            ProcessRewardConfig(progress_weight=0.9)


if __name__ == "__main__":
    unittest.main()
