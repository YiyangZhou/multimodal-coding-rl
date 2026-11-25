"""Validate opt-in original-history capture before native workers start."""

from pathlib import Path


def original_history_root(config, environ):
    value = environ.get("NEMO_GYM_EXPERIMENTAL_ORIGINAL_TOOL_HISTORY", "")
    if not value:
        return None
    root = Path(value)
    if not root.is_absolute():
        raise ValueError("Original tool history needs an absolute shared host directory")
    env = config.get("env", {})
    capture = env.get("nemo_gym", {}).get("token_id_capture", {})
    if not (
        env.get("should_mask_flagged_samples") is True
        and capture.get("enabled") is True
        and capture.get("rebuild_response") is False
        and capture.get("fail_fast_poisoned_rollouts") is True
        and config.get("token_capture", {}).get("enabled") is True
    ):
        raise ValueError("Original history requires external token capture, failure masking and fail-fast")
    # The native bridge sets external_staging when token_capture is enabled.
    if capture.get("external_staging") is False:
        raise ValueError("Original history cannot disable external staging")
    gym_root = environ.get("NEMO_GYM_ROOT", "")
    if not gym_root or not Path(gym_root).is_absolute():
        raise ValueError("Set an absolute NEMO_GYM_ROOT for original-history patch verification")
    return root
