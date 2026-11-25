"""Read the pinned Vision2Web directory layout without executing dataset files.

The public prototype/resource manifest is separate from private workflow rules.
Preparation creates no train split: upstream publishes evaluation tasks only.
"""
import hashlib
import json
from pathlib import Path, PurePosixPath

from .vision_workflows import parse_workflows

LEVELS = ("webpage", "frontend", "website")
MAX_TEXT_BYTES = 2 * 1024 * 1024


class MissingPrototypeError(ValueError):
    def __init__(self, relative):
        super().__init__("A workflow prototype has no exact upstream .jpg target: " + relative)
        self.relative = relative


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_file(root, relative):
    if not isinstance(relative, str):
        raise ValueError("Asset path must be text")
    path = PurePosixPath(relative)
    if (not path.parts or path.is_absolute() or ".." in path.parts or str(path) != relative
            or "\\" in relative or any(ord(c) < 32 for c in relative)):
        raise ValueError("Invalid dataset asset path")
    current = root
    for part in path.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("Dataset assets cannot be symlinks")
    if not current.is_file():
        raise ValueError("Dataset asset is missing or not a regular file")
    return current


def inventory(root, directory):
    base = root / directory
    if base.is_symlink() or not base.is_dir():
        raise ValueError(f"Missing or symlinked {directory} directory")
    records = []
    for path in sorted(base.rglob("*")):
        if path.is_symlink():
            raise ValueError("Dataset asset tree contains a symlink")
        if path.is_dir():
            continue
        relative = path.relative_to(root).as_posix()
        path = checked_file(root, relative)
        records.append({"path": relative, "bytes": path.stat().st_size, "sha256": file_digest(path)})
    return records


def read_text(root, name):
    path = checked_file(root, name)
    if path.stat().st_size > MAX_TEXT_BYTES:
        raise ValueError("Dataset text exceeds its size budget")
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError("Dataset text is empty")
    return text


def prepare_project(root, level):
    root = Path(root).resolve(strict=True)
    if level not in LEVELS or not root.is_dir():
        raise ValueError("Unknown Vision2Web level or project directory")
    private = json.loads(read_text(root, "workflow.json"))
    workflows = parse_workflows(private)
    prototypes = inventory(root, "prototypes")
    if not prototypes or len(prototypes) > 128:
        raise ValueError("Expected one to 128 prototype images")
    from PIL import Image

    for record in prototypes:
        path = checked_file(root, record["path"])
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"} or record["bytes"] > 32 * 1024 * 1024:
            raise ValueError("Unsupported or oversized prototype image")
        with Image.open(path) as image:
            image.verify()
    names = {r["path"] for r in prototypes}
    # Pinned upstream visual_scorer.py resolves names to prototypes/{name}.jpg.
    references = []
    for workflow in workflows:
        for node in workflow.nodes:
            if node.kind != "visual":
                continue
            relative = "prototypes/" + node.payload["name"] + ".jpg"
            if relative not in names:
                raise MissingPrototypeError(relative)
            references.append({"workflow": workflow.index, "position": node.index,
                               "name": node.payload["name"], "path": relative,
                               "fullpage": node.payload["fullpage"]})
    if not references:
        raise ValueError("Vision2Web task has no visual checkpoint")
    resources = inventory(root, "resources") if (root / "resources").exists() or (root / "resources").is_symlink() else []
    requirement_file = {"frontend": "prompt.txt", "website": "prd.md"}.get(level)
    requirements = read_text(root, requirement_file) if requirement_file else (
        "Implement a complete static webpage faithfully matching all supplied prototype images. "
        "Use the supplied resource files where relevant."
    )
    manifest = {
        "schema_version": 1, "benchmark": "vision2web", "level": level,
        "task_id": f"vision2web/{level}/{root.name}", "project_root": str(root),
        "upstream_split": "test", "requirements": requirements,
        "public_assets": prototypes + resources,
        "private_workflows": private, "private_visual_references": references,
        "source_files": [{"path": name, "bytes": (root / name).stat().st_size,
                          "sha256": file_digest(root / name)}
                         for name in ("workflow.json", *((requirement_file,) if requirement_file else ()))],
    }
    return manifest


def verify_manifest(manifest):
    """Re-read exact inputs before use; reject drift instead of grading another task."""
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("Unknown Vision2Web manifest")
    if prepare_project(manifest["project_root"], manifest["level"]) != manifest:
        raise ValueError("Vision2Web assets, requirements or private tests changed after preparation")
    return manifest


def public_input(manifest):
    """Single-turn content: requirements, labeled prototypes and asset names only."""
    from .tasks import image_uri

    verify_manifest(manifest)
    root = Path(manifest["project_root"])
    content = [{"type": "input_text", "text": manifest["requirements"]}]
    for record in manifest["public_assets"]:
        if record["path"].startswith("prototypes/"):
            content.extend([
                {"type": "input_text", "text": record["path"]},
                {"type": "input_image", "image_url": image_uri(str(root / record["path"])), "detail": "auto"},
            ])
    resources = [r["path"] for r in manifest["public_assets"] if r["path"].startswith("resources/")]
    content.append({"type": "input_text", "text": "Resource files available at execution time in /workspace:\n"
                    + json.dumps(resources, ensure_ascii=False)})
    return content
