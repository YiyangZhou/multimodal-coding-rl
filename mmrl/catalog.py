from dataclasses import dataclass

from .judge_contract import RUBRICS


@dataclass(frozen=True)
class Benchmark:
    name: str
    language: str
    artifact: str
    multi_fraction: float
    rationale: str
    rubric: str
    routes: tuple[str, ...] = ("single", "opencode", "codex", "hermes")


BENCHMARKS = {
    "vision2web": Benchmark(
        "Vision2Web", "json", "answer.json", 0.0,
        "Generate a complete project once; verify all prescribed UI workflows and visual states.",
        RUBRICS['vision2web'], routes=("single",),
    ),
    "chartmimic": Benchmark(
        "ChartMimic", "python", "answer.py", 0.0,
        "Generate or customize a chart; compare with the private target using the pinned renderer.",
        RUBRICS['chartmimic'], routes=("single",),
    ),
    "3dcodebench": Benchmark(
        "3DCodeBench", "python", "answer.py", 0.75,
        "Generate a Blender scene; inspect four canonical views in the verifier.",
        RUBRICS['3dcodebench'],
    ),
    "manimbench_v1": Benchmark(
        "ManimBench-v1",
        "python",
        "answer.py",
        0.75,
        "Generate a complete Manim scene; verify its final animation against the private reference.",
        RUBRICS['manimbench_v1'],

    ),
    "chart2code": Benchmark(
        "Chart2Code",
        "python",
        "answer.py",
        0.5,
        "Render and inspect axes, labels and data; keep a direct-generation baseline.",
        RUBRICS['chart2code'],
    ),
    "mmsvg_img2svg": Benchmark(
        "MMSVG · Img2SVG",
        "svg",
        "answer.svg",
        0.5,
        "Render feedback helps geometry and detail; cap unnecessary revisions.",
        RUBRICS['mmsvg_img2svg'],
    ),
    "webmmu_code_edit": Benchmark(
        "WebMMU · Code Edit",
        "diff",
        "answer.diff",
        0.75,
        "Inspect source, apply a strict diff and test the changed page.",
        RUBRICS['webmmu_code_edit'],
    ),
    "webmmu_mockup2code": Benchmark(
        "WebMMU · Mockup2Code",
        "html",
        "answer.html",
        0.5,
        "Screenshot feedback helps layout; preserve the dataset's exact instruction.",
        RUBRICS['webmmu_mockup2code'],
    ),
    "image2struct_latex": Benchmark(
        "Image2Struct",
        "latex",
        "answer.tex",
        0.75,
        "Compiler errors and page previews provide actionable repair feedback.",
        RUBRICS['image2struct_latex'],
    ),
    "mmsvg_text2svg": Benchmark(
        "MMSVG · Text2SVG",
        "svg",
        "answer.svg",
        0.25,
        "Direct generation is efficient; use a smaller render-check branch without a reference image.",
        RUBRICS['mmsvg_text2svg'],
    ),
    "design2code_full": Benchmark(
        "Design2Code",
        "html",
        "answer.html",
        0.5,
        "Render and inspect layout, text and spacing against the reference screenshot.",
        RUBRICS['design2code_full'],
    ),
}
HARNESS_NAMES = ("opencode", "codex", "hermes")
LEGACY_BENCHMARKS = (
    "chart2code", "mmsvg_img2svg", "webmmu_code_edit", "webmmu_mockup2code",
    "image2struct_latex", "mmsvg_text2svg", "design2code_full",
)
