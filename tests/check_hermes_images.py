"""Exercise pinned Hermes tool dispatch and request serialization, without inference."""

import base64
import io
import json
import tempfile
from pathlib import Path
from unittest.mock import patch


def main():
    from PIL import Image
    from openai.types.chat import ChatCompletionMessage
    from mmrl.hermes_runner import install

    install()
    install()  # Repeated setup must not duplicate observations or stack wrappers.
    from run_agent import AIAgent
    from tools.registry import registry

    assert registry.is_toolset_available("vision")
    with patch(
        "socket.socket.connect", side_effect=AssertionError("Unexpected network access")
    ):
        agent = AIAgent(
            base_url="https://capture.example.invalid/v1",
            api_key="test-only",
            model="policy",
            provider="custom",
            enabled_toolsets=["terminal", "file", "vision"],
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            persist_session=False,
            save_trajectories=False,
            use_streaming=False,
            tool_delay=0,
        )
        assert "vision_analyze" in agent.valid_tool_names
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "preview.png"
            messages = [{"role": "user", "content": "Inspect and improve this image."}]

            def call(call_id, image_path):
                response = ChatCompletionMessage(
                    role="assistant",
                    tool_calls=[
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": "vision_analyze",
                                "arguments": json.dumps(
                                    {
                                        "image_url": str(image_path),
                                        "question": "What changed?",
                                    }
                                ),
                            },
                        }
                    ],
                )
                messages.append(response.model_dump(exclude_none=True))
                agent._execute_tool_calls(response, messages, "image-probe", 1)

            Image.new("RGB", (12, 12), "red").save(path)
            call("first", path)
            first = messages[-1]["content"][1]["image_url"]["url"]
            assert messages[-2]["role"] == "tool"
            Image.new("RGB", (12, 12), "blue").save(path)
            call("second", path)
            second = messages[-1]["content"][1]["image_url"]["url"]
            assert first != second
            assert messages[3]["content"][1]["image_url"]["url"] == first
            for uri, expected in ((first, (255, 0, 0)), (second, (0, 0, 255))):
                with Image.open(
                    io.BytesIO(base64.b64decode(uri.split(",", 1)[1]))
                ) as image:
                    assert image.getpixel((0, 0)) == expected

            # A missing file stays a tool failure, never an invented image.
            call("missing", Path(tmp) / "missing.png")
            assert messages[-1]["role"] == "tool"
            assert "error" in json.loads(messages[-1]["content"])
            # Native interrupts remain effective.
            agent._interrupt_requested = True
            call("cancelled", path)
            assert messages[-1]["role"] == "tool"
            assert "cancelled" in messages[-1]["content"]
            agent._interrupt_requested = False

            request = agent._build_api_kwargs(messages)
            images = [
                part["image_url"]["url"]
                for msg in request["messages"]
                if isinstance(msg.get("content"), list)
                for part in msg["content"]
                if part.get("type") == "image_url"
            ]
            assert images == [first, second]
    print(
        "Native Hermes image tool, immutable snapshots, errors, interrupts and policy request: PASS"
    )
    print("No model inference or GPU alignment is claimed by this test.")


if __name__ == "__main__":
    main()
