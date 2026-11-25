from copy import deepcopy
from pathlib import Path
import unittest

from mmrl.history_config import original_history_root


class HistoryConfigurationTest(unittest.TestCase):
    def setUp(self):
        self.config = {
            "env": {"should_mask_flagged_samples": True, "nemo_gym": {
                "token_id_capture": {"enabled": True, "rebuild_response": False,
                                     "fail_fast_poisoned_rollouts": True}}},
            "token_capture": {"enabled": True},
        }
        self.env = {"NEMO_GYM_EXPERIMENTAL_ORIGINAL_TOOL_HISTORY": "/shared/run/history",
                    "NEMO_GYM_ROOT": "/shared/Gym"}

    def test_disabled_does_not_require_new_configuration(self):
        self.assertIsNone(original_history_root({}, {}))

    def test_native_bridge_can_supply_external_staging(self):
        self.assertEqual(original_history_root(self.config, self.env), Path("/shared/run/history"))

    def test_capture_protections_cannot_be_disabled(self):
        for key, value in (("enabled", False), ("rebuild_response", True),
                           ("fail_fast_poisoned_rollouts", False), ("external_staging", False)):
            config = deepcopy(self.config)
            config["env"]["nemo_gym"]["token_id_capture"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                original_history_root(config, self.env)
        for node, key in (("env", "should_mask_flagged_samples"), ("token_capture", "enabled")):
            config = deepcopy(self.config); config[node][key] = False
            with self.subTest(node=node), self.assertRaises(ValueError):
                original_history_root(config, self.env)

    def test_absolute_history_and_source_paths_required(self):
        for key in self.env:
            env = dict(self.env, **{key: "relative"})
            with self.subTest(key=key), self.assertRaises(ValueError):
                original_history_root(self.config, env)
        env = dict(self.env); del env["NEMO_GYM_ROOT"]
        with self.assertRaises(ValueError): original_history_root(self.config, env)


if __name__ == "__main__":
    unittest.main()
