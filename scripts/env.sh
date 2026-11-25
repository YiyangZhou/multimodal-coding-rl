#!/usr/bin/env bash
# Source this after activating the existing, locked NeMo RL / Gym environment.
mmrl_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$mmrl_repo${NEMO_GYM_ROOT:+:$NEMO_GYM_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export NEMO_GYM_EXTRA_ROOTS="$mmrl_repo/integrations/nemo_gym${NEMO_GYM_ROOT:+:$NEMO_GYM_ROOT}${NEMO_GYM_EXTRA_ROOTS:+:$NEMO_GYM_EXTRA_ROOTS}"
