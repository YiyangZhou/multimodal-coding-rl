"""Private, local evidence for one verifier invocation."""

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

HARNESS_FAILURE_CODES = frozenset({
    "codex_timeout", "codex_nonzero_exit", "codex_launch_failure",
    "codex_turn_incomplete_or_failed",
})


def safe_failure_markers(message):
    """Diagnostic hints only; persist fixed labels, never exception content."""
    if not isinstance(message, str):
        return []
    text = message[:8192].lower()
    patterns = {
        "context_limit": ("maximum context length", "context_length_exceeded", "max_model_len", "context window exceeded"),
        "capture_failed": ("rollout_capture_failed", "unresolved_parent", "mediacapturerejected"),
        "timeout": ("timed out", "timeouterror", "timeout exceeded"),
        "rate_limit": ("rate_limit_exceeded", "ratelimiterror", "too many requests"),
        "connection_failure": ("connection refused", "apiconnectionerror", "connection reset"),
        "budget_exceeded": ("budget_exceeded", "insufficient_quota"),
    }
    labels = {label for label, needles in patterns.items() if any(n in text for n in needles)}
    if "sandboxapiexception" in text and "failed to run command" in text:
        match = re.search(r"status code:\s*(410|502)\b", text)
        if match:
            labels.add("sandbox_http_" + match.group(1))
    return sorted(labels)


def safe_exit_diagnostics(raw):
    """Keep only bounded exit codes and known error categories; never raw logs."""
    if not isinstance(raw, str) or len(raw) > 8192:
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict):
        return None
    code = value.get("returncode")
    if type(code) is not int or not -255 <= code <= 255:
        return None
    statuses = value.get("http_statuses", [])
    markers = value.get("markers", [])
    if not isinstance(statuses, list) or not isinstance(markers, list):
        return None
    allowed = {
        "stream_disconnected", "timeout", "context_limit", "rate_limit",
        "connection_refused", "invalid_response",
    }
    return {
        "returncode": code,
        "http_statuses": sorted({s for s in statuses if type(s) is int and 400 <= s <= 599})[:16],
        "markers": sorted({m for m in markers if isinstance(m, str) and m in allowed}),
    }


def safe_terminal_metadata(route, metadata):
    """Preserve only terminal fields actually exposed by the configured adapter.

    completed is not exposed as Hermes metadata; response_status already records
    the adapter outcome. OpenCode exposes no root finish/exitcode in this schema.
    """
    if route != "hermes" or not isinstance(metadata, dict):
        return None
    result = {}
    for key in ("partial", "failed"):
        value = metadata.get(key)
        if type(value) is str and value in ("true", "false"):
            result[key] = value == "true"
    return result or None



def infrastructure_failure_code(reason):
    """Persist bounded categories only; never arbitrary exception text."""
    exact = {
        "Harness or model request failed; no valid task score": "upstream_request_failed",
        "All configured judge credentials failed authentication": "judge_all_credentials_rejected",
        "Configure OPENAI_API_KEYS or OPENAI_API_KEY": "judge_credentials_missing",
        "Quality judge transport failure; exclude/retry this measurement": "judge_transport",
        "Quality judge returned a non-JSON response": "judge_non_json",
        "Quality judge returned an invalid score": "judge_invalid_score",
        "Invalid image bytes from renderer or dataset": "invalid_image_bytes",
    }
    if not isinstance(reason, str):
        return "other_infrastructure"
    if reason in exact:
        return exact[reason]
    match = re.fullmatch(r"Quality judge HTTP ([1-5][0-9]{2}); exclude/retry this measurement", reason)
    return "judge_http_" + match.group(1) if match else "other_infrastructure"

class VerificationAudit:
    def __init__(self, directory, *, task, route, response_id, status, failure_code=None, rollout_id=None, exit_diagnostics=None, capture_media=False, terminal_metadata=None):
        self.capture_media = capture_media
        self.path = Path(directory) / uuid4().hex
        self.path.mkdir(mode=0o700, parents=True, exist_ok=False)
        self.record = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "benchmark": task.benchmark,
            "task_id": task.task_id,
            "prompt": task.prompt,
            "capture_media": capture_media,
            "route": route,
            "response_id": response_id,
            "rollout_id": rollout_id,
            "response_status": status,
            "files": {},
            "complete": False,
        }
        if failure_code is not None:
            self.record["harness_failure_code"] = (
                failure_code if isinstance(failure_code, str)
                and failure_code in HARNESS_FAILURE_CODES else "unclassified"
            )
            markers = safe_failure_markers(failure_code)
            if markers:
                self.record["harness_error_markers"] = markers
        diagnostic = safe_exit_diagnostics(exit_diagnostics)
        if route == "codex" and status == "failed" and diagnostic is not None:
            self.record["harness_exit_diagnostics"] = diagnostic
        terminal = safe_terminal_metadata(route, terminal_metadata)
        if terminal is not None:
            self.record["upstream_terminal"] = terminal
        self._save_record()

    def write(self, name, content):
        visual = re.fullmatch(r"workflow-(0|[1-9][0-9]{0,2})-visual-(0|[1-9][0-9]{0,2})\.png", name)
        vision_png = visual is not None and all(int(index) < 128 for index in visual.groups())
        if name not in {
            "answer.txt", "render_source.txt", "render.png", "final_message.txt",
            "judge_reason.txt",
            "original_render.png", "target_render.png",
            "animation.json", "reference_animation.json", "reference_render.png",
            "object_views.json", "reference_object_views.json",
            "deployment.json", "workflows.json", "vision_reward.json",
        } and not vision_png:
            raise ValueError("Unknown verification artifact")
        data = content.encode() if isinstance(content, str) else content
        with (self.path / name).open("xb") as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write(data)
        self.record["files"][name] = {
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }

    def finish(self, result):
        # Do not copy requests, sandbox descriptors, headers, or arbitrary
        # exception messages (which may contain authenticated URLs).
        self.record["result"] = {
            key: result[key]
            for key in (
                "reward", "execution", "task_fidelity", "visual_fidelity",
                "failure_kind", "mask_sample",
            )
            if key in result
        }
        reason = result.get("failure_reason")
        if result.get("failure_kind") == "mm_coding:infrastructure":
            self.record["result"]["infrastructure_failure_code"] = infrastructure_failure_code(reason)
        if isinstance(reason, str) and reason in {
            "artifact_missing_or_not_regular", "artifact_symlink",
            "artifact_oversize", "artifact_size_probe_failed", "artifact_probe_failed",
        }:
            self.record["result"]["artifact_failure_code"] = reason
        self.record["complete"] = True
        self._save_record()
        # Small immutable lookup records avoid scanning all historical rollouts.
        rid = self.record.get("rollout_id")
        if isinstance(rid, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,256}", rid):
            lookup = self.path.parent / "by-rollout" / rid
            lookup.mkdir(parents=True, exist_ok=True, mode=0o700)
            target = lookup / (self.path.name + ".json")
            temporary = target.with_suffix(".tmp")
            with temporary.open("w") as stream:
                os.fchmod(stream.fileno(), 0o600)
                json.dump(self.record, stream, allow_nan=False)
            temporary.replace(target)


    def _save_record(self):
        temporary = self.path / "record.json.tmp"
        with temporary.open("w") as stream:
            os.fchmod(stream.fileno(), 0o600)
            json.dump(self.record, stream, indent=2, allow_nan=False)
            stream.write("\n")
        temporary.replace(self.path / "record.json")
