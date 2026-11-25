import asyncio
import base64
import json
import math
import os
import urllib.error
import urllib.request
from dataclasses import dataclass

from .catalog import BENCHMARKS
from .judge_contract import JUDGE_INSTRUCTIONS


class InfrastructureError(RuntimeError):
    """No valid model-quality measurement was produced."""


class JudgeAuthenticationError(InfrastructureError):
    """The selected judge credential was rejected."""


def image_data_uri(data):
    from .judge_images import bounded_image_bytes

    try:
        data = bounded_image_bytes(data)
    except ValueError as exc:
        raise InfrastructureError("Invalid image bytes from renderer or dataset") from exc
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        mime = "image/png"
    elif data.startswith(b"\xff\xd8\xff"):
        mime = "image/jpeg"
    elif data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        mime = "image/webp"
    else:
        raise InfrastructureError("Invalid image bytes from renderer or dataset")
    return f"data:{mime};base64," + base64.b64encode(data).decode()


@dataclass(frozen=True)
class Quality:
    task_fidelity: float
    visual_fidelity: float
    reason: str = ""

    def __post_init__(self):
        for value in (self.task_fidelity, self.visual_fidelity):
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError("Judge dimensions must be finite numbers in [0, 1]")
        if not isinstance(self.reason, str) or len(self.reason) > 4000:
            raise ValueError("Judge reason must be bounded text")

    @classmethod
    def from_response(cls, response):
        if not isinstance(response, dict) or set(response) != {
            "quality", "task_fidelity", "visual_fidelity", "reason"
        }:
            raise ValueError("Expected the original four-field judge contract")
        result = cls(response["task_fidelity"], response["visual_fidelity"], response["reason"])
        quality = response["quality"]
        if (isinstance(quality, bool) or not isinstance(quality, (int, float))
                or not math.isfinite(quality) or not 0 <= quality <= 1
                or not math.isclose(quality, (result.task_fidelity + result.visual_fidelity) / 2,
                                    abs_tol=0.005)
                or not result.reason):
            raise ValueError("Invalid quality average or empty judge reason")
        return result

    @property
    def reward(self):
        return 0.2 + 0.8 * (self.task_fidelity + self.visual_fidelity) / 2


class QualityJudge:
    """Trusted-side judge. Credentials never enter agent sandboxes."""

    def __init__(self, concurrency=8):
        if (
            isinstance(concurrency, bool)
            or not isinstance(concurrency, int)
            or concurrency < 1
        ):
            raise ValueError("Judge concurrency must be a positive integer")
        self._sem = asyncio.Semaphore(concurrency)
        keys = os.environ.get("OPENAI_API_KEYS", os.environ.get("OPENAI_API_KEY", ""))
        self._keys = tuple(
            dict.fromkeys(k.strip() for k in keys.split(",") if k.strip())
        )
        if not self._keys:
            raise InfrastructureError("Configure OPENAI_API_KEYS or OPENAI_API_KEY")
        self.model = os.environ.get("MMRL_JUDGE_MODEL", "").strip()
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        if not self.model:
            raise InfrastructureError("Configure MMRL_JUDGE_MODEL")
        self.responses_url = base_url + "/responses"
        self._next = 0
        self._disabled_keys = set()

    async def grade(self, task, candidate, png, reference_images, *,
                    original_render=None, target_render=None):
        async with self._sem:
            for _ in range(len(self._keys)):
                available = [k for k in self._keys if k not in self._disabled_keys]
                if not available:
                    break
                key = available[self._next % len(available)]
                self._next += 1
                try:
                    return await asyncio.to_thread(
                        self._grade, key, task, candidate, png, reference_images,
                        original_render, target_render,
                    )
                except JudgeAuthenticationError:
                    self._disabled_keys.add(key)
            raise InfrastructureError("All configured judge credentials failed authentication")

    def _grade(self, key, task, candidate, png, reference_images,
               original_render=None, target_render=None):
        content = [
            {
                "type": "input_text",
                "text": json.dumps(
                    {
                        "task": task.prompt,
                        "benchmark": task.benchmark,
                        "semantic_prompt": task.semantic_prompt,
                        "rubric": BENCHMARKS[task.benchmark].rubric,
                        "candidate_source": candidate,
                        "original_source": task.original_source,
                        "reference_patch": task.reference_patch,
                        "reference_code_or_patch": task.ground_truth or task.reference_patch,
                        "reference_render_available": (
                            target_render is not None if task.benchmark == "webmmu_code_edit"
                            else bool(reference_images)
                        ),
                    }
                ),
            }
        ]
        def add_image(label, data):
            content.extend([
                {"type": "input_text", "text": label},
                {"type": "input_image", "image_url": image_data_uri(data)},
            ])

        for im in reference_images:
            add_image("ORIGINAL input screenshot BEFORE the requested edit (not the target)"
                      if task.benchmark == "webmmu_code_edit" else "REFERENCE target rendering", im)
        if original_render is not None:
            add_image("ORIGINAL page rendering before the edit", original_render)
        if target_render is not None:
            add_image("REFERENCE target rendering", target_render)
        add_image("CANDIDATE rendering to grade", png)
        payload = {
            "model": self.model,
            "instructions": JUDGE_INSTRUCTIONS + BENCHMARKS[task.benchmark].rubric,
            "input": [{"role": "user", "content": content}],
            "max_output_tokens": 2048,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "coding_quality",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            **{
                            k: {"type": "number", "minimum": 0, "maximum": 1}
                            for k in ("quality", "task_fidelity", "visual_fidelity")
                            },
                            "reason": {"type": "string", "minLength": 1, "maxLength": 4000},
                        },
                        "required": ["quality", "task_fidelity", "visual_fidelity", "reason"],
                    },
                }
            },
        }
        request = urllib.request.Request(
            self.responses_url,
            json.dumps(payload).encode(),
            {"Authorization": "Bearer " + key, "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                result = json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code == 401:
                raise JudgeAuthenticationError("Judge credential rejected (HTTP 401)") from None
            raise InfrastructureError(
                f"Quality judge HTTP {exc.code}; exclude/retry this measurement"
            ) from None
        except (urllib.error.URLError, TimeoutError) as exc:
            raise InfrastructureError(
                "Quality judge transport failure; exclude/retry this measurement"
            ) from exc
        except json.JSONDecodeError as exc:
            raise InfrastructureError("Quality judge returned a non-JSON response") from exc
        try:
            if not isinstance(result, dict):
                raise ValueError("Expected an Quality judge response object")
            if result.get("status") not in (None, "completed"):
                raise ValueError("Incomplete judge response")
            text = "".join(
                c["text"]
                for o in result["output"]
                if o["type"] == "message"
                for c in o["content"]
                if c["type"] == "output_text"
            )
            return Quality.from_response(json.loads(text))
        except (KeyError, TypeError, ValueError) as exc:
            raise InfrastructureError("Quality judge returned an invalid score") from exc
