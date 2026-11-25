"""CPU-only Gym serialization -> pinned native loss-mask consumer regression."""

import json
from mmrl.gym_app import VerifyResponse
from nemo_gym.openai_utils import NeMoGymResponse
from responses_api_agents.harness_agent.app import HarnessAgentVerifyResponse
from responses_api_agents.simple_agent.app import SimpleAgentVerifyResponse
from nemo_rl.experience.rollouts import _mask_sample_flags

response = NeMoGymResponse(
    id="resp_mask_probe",
    created_at=1.0,
    model="policy",
    object="response",
    status="completed",
    output=[],
    tool_choice="auto",
    tools=[],
    parallel_tool_calls=True,
)
rows = []
for wrapper in (SimpleAgentVerifyResponse, HarnessAgentVerifyResponse):
    for masked in (False, True):
        result = VerifyResponse(
            responses_create_params={"input": []},
            response=response,
            reward=0.0,
            mask_sample=masked,
            instance_config={"mask_sample": not masked, "probe": True},
        )
        row = wrapper.model_validate(result.model_dump(mode="json")).model_dump(
            mode="json"
        )
        rows.append(row)
flags = _mask_sample_flags(rows).tolist()
assert flags == [False, True, False, True], flags
assert all(row["instance_config"]["probe"] for row in rows)
print(
    json.dumps({"native_loss_mask_consumer_passed": True, "flags": flags}), flush=True
)
