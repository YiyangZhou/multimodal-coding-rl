"""One policy response can describe a complete runnable web project."""
import io
import json
from pathlib import PurePosixPath
import tarfile


def _unique_object(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def parse_project(text, *, max_bytes=4 * 1024 * 1024, max_files=256):
    """Validate text files without executing or writing a candidate on the host.

    Dataset assets and private verifier inputs cannot be replaced by the answer.
    Files are later extracted only into a fresh execution sandbox.
    """
    if not isinstance(text, str) or len(text.encode("utf-8")) > max_bytes:
        raise ValueError("Project response exceeds its byte budget")
    data = json.loads(text, object_pairs_hook=_unique_object)
    if not isinstance(data, dict) or set(data) != {"files"} or not isinstance(data["files"], list):
        raise ValueError("Expected an object containing a files list")
    if not 1 <= len(data["files"]) <= max_files:
        raise ValueError("Project file count is outside its budget")
    files = {}
    total = 0
    for item in data["files"]:
        if not isinstance(item, dict) or set(item) != {"path", "content"}:
            raise ValueError("Each project file needs path and content")
        name, content = item["path"], item["content"]
        if not isinstance(name, str) or not isinstance(content, str):
            raise ValueError("Project file names and contents must be text")
        path = PurePosixPath(name)
        if (not name or not path.parts or name != str(path) or path.is_absolute() or ".." in path.parts
                or "\\" in name or any(ord(c) < 32 for c in name) or len(name) > 240
                or path.parts[0] in {"resources", "prototypes", ".git", ".mmrl"}
                or name in {"workflow.json", "prompt.txt", "prd.md"} or name in files
                or "\x00" in content):
            raise ValueError("Invalid, duplicated or reserved project file path")
        encoded = content.encode("utf-8")
        total += len(encoded)
        if total > max_bytes:
            raise ValueError("Project content exceeds its byte budget")
        files[name] = encoded
    if not files.get("start.sh", b"").strip():
        raise ValueError("A complete project requires start.sh")
    for name in files:
        if any(str(parent) in files for parent in PurePosixPath(name).parents if str(parent) != "."):
            raise ValueError("Project path is both a file and a directory")
    return files


def project_archive(files):
    """Create regular files only; supplied paths must already pass parse_project."""
    canonical = parse_project(json.dumps({"files": [
        {"path": name, "content": value.decode("utf-8")} for name, value in files.items()
    ]}))
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz", format=tarfile.PAX_FORMAT) as archive:
        for name, value in sorted(canonical.items()):
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.mtime = len(value), 0o644, 0
            archive.addfile(info, io.BytesIO(value))
    return output.getvalue()
