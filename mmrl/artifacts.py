import json
import re


def extract_answer(text, language):
    """Use final answer only; never salvage an unfinished thinking/code fragment."""
    text = text.strip()
    if language == "json":
        if text.startswith("<think>"):
            match = re.fullmatch(r"<think>[\s\S]*?</think>\s*([\s\S]*)", text)
            if not match:
                raise ValueError("Unfinished thinking")
            text = match[1].strip()
        fence = re.fullmatch(r"```json\s*\n([\s\S]*?)\n\s*```", text)
        if fence:
            text = fence[1]
        # Complete JSON can legitimately contain Markdown fences or XML tags in
        # a source file's string value. Never mistake those for rollout boundaries.
        json.loads(text)
        return text
    if "<think>" in text:
        match = re.fullmatch(r"<think>[\s\S]*?</think>\s*([\s\S]*)", text)
        if not match:
            raise ValueError("Unfinished thinking")
        text = match[1].strip()
    fence = re.fullmatch(r"```([\w+-]*)\s*\n([\s\S]*?)\n\s*```", text)
    if fence:
        aliases = {
            "python": {"python", "py"},
            "svg": {"svg", "xml"},
            "html": {"html"},
            "diff": {"diff", "patch"},
            "latex": {"latex", "tex"},
        }
        if fence[1].lower() not in aliases[language]:
            raise ValueError("Wrong code fence")
        text = fence[2]
    if not text.strip() or "```" in text or "<think>" in text or "</think>" in text:
        raise ValueError("Expected one complete artifact")
    return text


def apply_diff(original, patch):
    """Apply a single-file unified diff with exact context and counts."""
    lines, out, cursor = original.splitlines(keepends=True), [], 0
    parts = patch.splitlines(keepends=True)
    i = 0
    if len(parts) >= 2 and parts[0].startswith("--- ") and parts[1].startswith("+++ "):
        i = 2
    hunks = 0
    while i < len(parts):
        match = re.fullmatch(
            r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@[^\n]*\n?", parts[i]
        )
        if not match:
            raise ValueError("Malformed unified diff")
        old_start, old_count, new_start, new_count = (
            int(match[1]),
            int(match[2] or 1),
            int(match[3]),
            int(match[4] or 1),
        )
        start = old_start if old_count == 0 else old_start - 1
        if not cursor <= start <= len(lines):
            raise ValueError("Overlapping or out-of-range hunk")
        out.extend(lines[cursor:start])
        cursor = start
        if (new_start if new_count == 0 else new_start - 1) != len(out):
            raise ValueError("New hunk offset mismatch")
        i += 1
        old_seen = new_seen = 0
        hunks += 1
        while i < len(parts) and not parts[i].startswith("@@ "):
            line = parts[i]
            i += 1
            if not line or line[0] not in " +-":
                raise ValueError("Invalid diff content")
            content = line[1:]
            if i < len(parts) and parts[i].startswith("\\ No newline at end of file"):
                content = content.removesuffix("\n")
                i += 1
            if line[0] in " -":
                if cursor >= len(lines) or lines[cursor] != content:
                    raise ValueError("Diff does not match original exactly")
                cursor += 1
                old_seen += 1
            if line[0] in " +":
                out.append(content)
                new_seen += 1
        if (old_seen, new_seen) != (old_count, new_count):
            raise ValueError("Hunk length mismatch")
    if not hunks:
        raise ValueError("Empty diff")
    return "".join(out + lines[cursor:])
