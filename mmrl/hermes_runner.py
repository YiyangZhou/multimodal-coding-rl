"""Run the pinned Gym Hermes adapter with images returned to the policy itself.

Hermes' stock vision_analyze delegates to an auxiliary model. Here the native
tool loop stays intact, but that tool snapshots a local image and returns its
pixels as the next user observation. No auxiliary inference is involved.
"""

import base64
import io
import json
import runpy
import sys
import uuid
from pathlib import Path

_installed = False


class ImageFeedback:
    def __init__(self):
        self.pending = {}

    def capture(self, args, **kwargs):
        from PIL import Image

        source = Path(args["image_url"])
        with source.open("rb") as stream:
            data = stream.read(16 * 1024 * 1024 + 1)
        if len(data) > 16 * 1024 * 1024:
            raise ValueError("Image exceeds the 16 MiB tool limit")
        with Image.open(io.BytesIO(data)) as picture:
            mime = Image.MIME.get(picture.format)
            picture.verify()
        if mime not in ("image/png", "image/jpeg", "image/webp"):
            raise ValueError("Render the image to PNG, JPEG or WebP first")
        observation = uuid.uuid4().hex
        self.pending[observation] = [
            {"type": "text", "text": f"Image from {source}: {args['question']}"},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime};base64," + base64.b64encode(data).decode()
                },
            },
        ]
        return json.dumps({"image_observation": observation, "source": str(source)})

    def append(self, messages, start, calls):
        vision_ids = {c.id for c in calls if c.function.name == "vision_analyze"}
        content = []
        for message in messages[start:]:
            if (
                message.get("role") != "tool"
                or message.get("tool_call_id") not in vision_ids
            ):
                continue
            try:
                result = json.loads(message["content"])
            except (json.JSONDecodeError, TypeError):
                continue  # Native cancellation text, not a successful image read.
            if isinstance(result, dict) and "image_observation" in result:
                content.extend(self.pending.pop(result["image_observation"]))
        if content:
            messages.append({"role": "user", "content": content})


def install():
    global _installed
    if _installed:
        return
    import run_agent
    from tools.registry import registry

    feedback = ImageFeedback()
    registry.deregister("vision_analyze")
    registry.register(
        name="vision_analyze",
        toolset="vision",
        schema={
            "name": "vision_analyze",
            "description": "View a local image. Its pixels are returned to you; inspect it yourself.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_url": {"type": "string", "description": "Local image path"},
                    "question": {"type": "string", "description": "What to inspect"},
                },
                "required": ["image_url", "question"],
            },
        },
        handler=feedback.capture,
    )

    class VisualAgent(run_agent.AIAgent):
        def _execute_tool_calls(
            self, assistant_message, messages, effective_task_id, api_call_count=0
        ):
            start = len(messages)
            super()._execute_tool_calls(
                assistant_message, messages, effective_task_id, api_call_count
            )
            feedback.append(messages, start, assistant_message.tool_calls)

    run_agent.AIAgent = VisualAgent
    _installed = True


if __name__ == "__main__":
    sys.argv = sys.argv[1:]
    # Gym's runner mounts its own snapshot here. Import that same copy, and
    # defer Hermes initialization until Gym has set HERMES_HOME and terminal
    # settings in model_post_init (tools cache some settings at import time).
    sys.path.insert(0, str(Path(sys.argv[0]).resolve().parent / "gym_mount"))
    from responses_api_agents.hermes_agent.app import HermesAgent

    original_create = HermesAgent._create_response

    async def create_with_images(self, *args, **kwargs):
        install()
        return await original_create(self, *args, **kwargs)

    HermesAgent._create_response = create_with_images
    runpy.run_path(sys.argv[0], run_name="__main__")
