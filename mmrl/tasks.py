import base64
import hashlib
import json
import mimetypes
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .catalog import BENCHMARKS, HARNESS_NAMES


@dataclass(frozen=True)
class Task:
    benchmark: str
    task_id: str
    prompt: str
    references: list[str] = field(default_factory=list)
    original_source: str = ""
    reference_patch: str = ""
    category: str = ""
    semantic_prompt: str = ""
    ground_truth: str = ""
    judge_references: list[str] = field(default_factory=list)
    vision_manifest: dict = field(default_factory=dict)

    def __post_init__(self):
        if (
            self.benchmark not in BENCHMARKS
            or not self.task_id
            or not self.prompt.strip()
        ):
            raise ValueError(
                "A task needs a known benchmark, stable ID and original prompt"
            )
        text_only = self.benchmark in {"mmsvg_text2svg", "manimbench_v1"}
        if self.vision_manifest and self.benchmark != "vision2web":
            raise ValueError("Vision2Web manifests are exclusive to Vision2Web tasks")
        if self.judge_references and self.benchmark != "chartmimic":
            raise ValueError("Separate target images are currently supported only for ChartMimic")
        if self.benchmark == "chartmimic":
            if self.category not in {"direct", "customized"}:
                raise ValueError("ChartMimic requires an explicit direct/customized task type")
            if len(self.judge_references) != 1:
                raise ValueError("ChartMimic requires one private target image, distinct from model input")
        if text_only and self.references:
            raise ValueError("Text-only tasks have no input reference image")
        if self.benchmark == "manimbench_v1" and not self.ground_truth.strip():
            raise ValueError("ManimBench requires private reference code for verification")
        if self.benchmark == "vision2web":
            from .vision_data import verify_manifest

            manifest = verify_manifest(self.vision_manifest)
            expected_images = [str(Path(manifest["project_root"]) / item["path"])
                               for item in manifest["public_assets"] if item["path"].startswith("prototypes/")]
            if (self.references != expected_images or self.prompt != manifest["requirements"]
                    or self.task_id != manifest["task_id"] or self.category != manifest["level"]
                    or self.ground_truth or self.original_source or self.reference_patch):
                raise ValueError("Vision2Web task does not match its verified project manifest")
        elif self.benchmark == "3dcodebench":
            if len(self.references) not in (0, 1, 2, 3, 4) or not self.ground_truth.strip():
                raise ValueError("3DCodeBench needs private reference code and zero to four input views")
        elif self.benchmark == "webmmu_code_edit":
            if not self.original_source or not self.reference_patch:
                raise ValueError(
                    "Code Edit requires original source and a reference diff"
                )
        elif not text_only and len(self.references) != 1:
            raise ValueError("Visual tasks require exactly one reference image")

    @classmethod
    def from_legacy_context(cls, context):
        return cls(
            benchmark=context["benchmark_id"],
            task_id=str(context["sample_id"]),
            prompt=context["prompt"],
            references=context["target_image_paths"],
            original_source=context.get("original_source", ""),
            reference_patch=context.get("ground_truth", "")
            if context["benchmark_id"] == "webmmu_code_edit"
            else "",
            category=context.get("category", ""),
            semantic_prompt=context.get("semantic_prompt", ""),
            ground_truth=context.get("ground_truth", ""),
        )


def image_uri(path):
    data = Path(path).read_bytes()
    if len(data) > 32 * 1024 * 1024:
        raise ValueError("Reference image exceeds 32 MiB")
    mime = mimetypes.guess_type(path)[0]
    if mime not in {"image/png", "image/jpeg", "image/webp"}:
        raise ValueError("Use PNG, JPEG or WebP reference images")
    return f"data:{mime};base64," + base64.b64encode(data).decode()


def route(task, mode, seed=42, epoch=0, harnesses=HARNESS_NAMES, multi_fraction=None):
    """Route before sibling expansion, never by reward or completion speed."""
    if mode not in {"single", "multi", "mixed"}:
        raise ValueError("mode must be single, multi or mixed")
    if (
        not harnesses
        or set(harnesses) - set(HARNESS_NAMES)
        or len(set(harnesses)) != len(harnesses)
    ):
        raise ValueError("Choose unique supported harnesses")
    fraction = (
        BENCHMARKS[task.benchmark].multi_fraction
        if multi_fraction is None
        else multi_fraction
    )
    if not 0 <= fraction <= 1:
        raise ValueError("multi_fraction must be in [0, 1]")
    digest = hashlib.sha256(
        f"{seed}:{epoch}:{task.benchmark}:{task.task_id}".encode()
    ).digest()
    use_multi = (
        mode == "multi"
        or mode == "mixed"
        and int.from_bytes(digest[:8], "big") / 2**64 < fraction
    )
    selected = (
        harnesses[int.from_bytes(digest[8:16], "big") % len(harnesses)]
        if use_multi
        else "single"
    )
    if selected not in BENCHMARKS[task.benchmark].routes:
        raise ValueError(f"{task.benchmark} does not support route {selected}")
    return selected


def reference_paths(references):
    """Tool-visible paths shared by row preparation and sandbox upload."""
    paths = []
    for index, reference in enumerate(references):
        suffix = Path(reference).suffix.lower()
        if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
            raise ValueError("Unsupported reference image format")
        stem = "reference" if len(references) == 1 else f"reference-{index:02d}"
        paths.append(f"/workspace/{stem}{suffix}")
    return paths


def gym_row(task, selected, max_output_tokens=32768, *, process_rewards=False):
    if selected not in ("single", *HARNESS_NAMES):
        raise ValueError("Unknown route")
    if selected not in BENCHMARKS[task.benchmark].routes:
        raise ValueError(f"{task.benchmark} does not support route {selected}")
    if not 1 <= max_output_tokens <= 262144:
        raise ValueError("Explicit output budget required (1..262144)")
    content = [{"type": "input_text", "text": task.prompt}]
    if task.benchmark == "vision2web":
        from .vision_data import public_input

        content = public_input(task.vision_manifest)
    elif selected == "single":
        if task.original_source and task.original_source not in task.prompt:
            content.append(
                {
                    "type": "input_text",
                    "text": "Original source to edit:\n" + task.original_source,
                }
            )
        content.extend(
            {"type": "input_image", "image_url": image_uri(p), "detail": "auto"}
            for p in task.references
        )
    elif task.references:
        content.append(
            {
                "type": "input_text",
                "text": "Task reference images: " + ", ".join(reference_paths(task.references))
                + ". Inspect all views with your image tool before solving.",
            }
        )
    messages = [{"role": "user", "content": content}]
    if task.benchmark == "vision2web":
        messages.insert(0, {
            "role": "system",
            "content": 'Return one complete runnable web project as JSON: '
                       '{"files":[{"path":"start.sh","content":"..."},'
                       '{"path":"...","content":"..."}]}. Use a json code block. '
                       'Paths are relative to /workspace. Include all source files and a self-contained '
                       'start.sh that installs dependencies, initializes any required database and '
                       'serves the app at http://localhost:3000. For full-stack tasks implement a real '
                       'backend and persistent data; do not replace these with a static mockup. '
                       'Dataset resources will be provided under /workspace/resources. Do not include '
                       'resource or prototype files in the JSON. You receive all prototypes as images; '
                       'prototype files are not provided to the running application. '
                       'Maximum 256 UTF-8 text files and 4 MiB of JSON.',
        })
    if task.benchmark == "chartmimic":
        messages.insert(0, {
            "role": "system",
            "content": "Implement the requested chart in Python with Matplotlib 3.8.4. "
                       "Return a complete Python program in a python code block. Save the "
                       "final chart as the single-page PDF /workspace/chart.pdf. Preserve "
                       "the requested dimensions, data and labels. The attached image is "
                       "the input to the task; apply any requested changes to it.",
        })
    if task.benchmark == "3dcodebench":
        messages.insert(0, {
            "role": "system",
            "content": "Implement the requested 3D object as a self-contained Blender 5.0 "
                       "Python script using bpy. Return the complete program in a python code "
                       "block. Clear the default scene and create only the requested object, "
                       "without a ground plane, backdrop or unrelated props. Leave geometry "
                       "untextured; do not reproduce colors, materials or surface textures. "
                       "The verifier supplies cameras and lights and renders four views. "
                       "Do not render, save a blend file or quit Blender. Attached images, "
                       "if any, depict the same object from different views.",
        })
    if task.benchmark == "manimbench_v1":
        messages.insert(0, {
            "role": "system",
            "content": "Implement the requested scene using Manim Community 0.19.0. "
                       "Return a complete Python program with the required imports and Scene "
                       "subclass(es), inside a python code block. The verifier renders every "
                       "scene with the Cairo renderer. Preserve the requested animation and timing.",
        })
    if selected != "single":
        artifact = BENCHMARKS[task.benchmark].artifact
        preview_instruction = (
            "After saving a code version, use the available preview tool from the mmrl MCP server "
            "to render it. Inspect the returned image with your image tool, then revise if needed. "
            if process_rewards else
            "Preview with `python /opt/mmrl/render.py "
            f"{task.benchmark} /workspace/{artifact} /workspace/preview.png`. "
            "You may inspect the preview with your image tool. "
        )
        if task.benchmark in {"3dcodebench", "manimbench_v1"}:
            runtime = (
                "Blender 5.0 is installed at /opt/blender/blender; bpy runs inside Blender, "
                "not the default Python interpreter. "
                if task.benchmark == "3dcodebench" else
                "Manim Community 0.19.0 is installed in /opt/manim; its interpreter is "
                "/opt/manim/bin/python. The provided preview command selects this runtime. "
            )
            preview_instruction += (
                runtime + "The preview entrypoint is already installed; use the provided preview interface "
                "rather than searching renderer or grader implementation files. "
                "Save a complete first version early, then use tool feedback to revise it. "
                "Keep the user's requirements and final artifact path unchanged. "
            )
        if task.benchmark == "3dcodebench":
            preview_instruction += (
                "Work artifact-first: write a complete initial /workspace/answer.py "
                "before exploratory environment inspection or pixel-analysis scripts. "
                "Use the provided reference and task to construct the model directly. "
                "Then run the supplied preview command and revise your saved program. "
                "Do not read renderer or grader source code; the runtime and preview "
                "interface above are sufficient. Preserve the requested geometry and "
                "details; this changes the work order, not the task requirements. "
            )
        messages.insert(
            0,
            {
                "role": "system",
                "content": (
                    f"Solve the user's task in /workspace. Save your final answer to /workspace/{artifact}. "
                    "Use shell and file tools to inspect and revise. "
                    + preview_instruction + "Finish with a final response. "
                    "For Code Edit, submit a unified diff against original.html. Do not change the task requirements."
                ),
            },
        )
    return {
        "agent_ref": {"type": "responses_api_agents", "name": f"mm_{selected}"},
        "task": asdict(task),
        "route": selected,
        "responses_create_params": {
            "model": "policy",
            "input": messages,
            "max_output_tokens": max_output_tokens,
            "temperature": 1.0,
            "tools": [],
        },
    }


def sibling_mixed_row(task, max_output_tokens=32768, *, process_rewards=False):
    """One dataset prompt; native expansion assigns two siblings to each route."""
    from .sibling_routes import ROUTES
    variants = {name: gym_row(task, name, max_output_tokens,
                             process_rewards=process_rewards) for name in ROUTES}
    row = dict(variants["single"])
    row["sibling_variants"] = variants
    return row


def write_rows(tasks, path, mode, *, max_output_tokens=32768, process_rewards=False, **kwargs):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as out:
        for task in tasks:
            if mode == "sibling-mixed":
                row = sibling_mixed_row(task, max_output_tokens,
                                        process_rewards=process_rewards)
            else:
                row = gym_row(task, route(task, mode, **kwargs),
                              max_output_tokens=max_output_tokens,
                              process_rewards=process_rewards)
            out.write(json.dumps(row) + "\n")
