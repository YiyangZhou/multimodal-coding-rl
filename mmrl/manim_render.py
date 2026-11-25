"""Render animation evidence INSIDE a disposable verification sandbox only."""

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess


class RuntimeUnavailable(RuntimeError):
    pass


def check_runtime(python):
    for name in (python, "ffmpeg", "ffprobe", "latex", "dvisvgm"):
        if not shutil.which(name):
            raise RuntimeUnavailable(f"Missing renderer dependency: {name}")
    try:
        probe = subprocess.run(
            [python, "-c", "import manim; assert manim.__version__ == '0.19.0'"],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeUnavailable("Manim runtime could not initialize") from exc
    if probe.returncode:
        raise RuntimeUnavailable("Manim CE0.19.0 runtime preflight failed")
    try:
        assets = subprocess.run(
            [python, "-c", "from manim_assets import install_asset_cache; install_asset_cache()"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeUnavailable("Pinned Manim asset preflight failed") from exc
    if assets.returncode or not Path(__file__).with_name("manim_entry.py").is_file():
        raise RuntimeUnavailable("Pinned Manim public assets or entrypoint unavailable")


def final_artifacts(media):
    """Intermediate movie fragments and TeX assets never count as final answers."""
    outputs = []
    for directory, suffix in (("videos", ".mp4"), ("images", ".png")):
        for path in sorted((media / directory).rglob("*" + suffix)):
            if "partial_movie_files" in path.parts or path.is_symlink():
                continue
            if path.stat().st_size and path.resolve().is_relative_to(media.resolve()):
                outputs.append(path)
    return outputs


def video_metadata(path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
         "stream=width,height,nb_read_frames,avg_frame_rate:format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, timeout=30, check=True,
    )
    data = json.loads(result.stdout)
    stream = data["streams"][0]
    duration = float(data["format"]["duration"])
    count = int(stream["nb_read_frames"])
    numerator, denominator = stream["avg_frame_rate"].split("/")
    if float(denominator) == 0:
        raise ValueError("Final video has an undefined frame rate")
    fps = float(numerator) / float(denominator)
    if not math.isfinite(duration) or duration <= 0 or min(stream["width"], stream["height"]) <= 0:
        raise ValueError("Invalid final video metadata")
    if count < 1 or not math.isfinite(fps) or fps <= 0:
        raise ValueError("Final video has no decodable frames")
    stream.update(frame_count=count, fps=fps)
    return duration, stream


def evidence_frames(artifacts, output, frames_per_video=12):
    from PIL import Image

    frames, scenes = [], []
    for index, artifact in enumerate(artifacts):
        if artifact.stat().st_size > 128 * 1024 * 1024:
            raise ValueError("Scene output exceeds 128 MiB verification budget")
        if artifact.suffix == ".png":
            with Image.open(artifact) as image:
                image.load()
                if image.width * image.height > 32_000_000:
                    raise ValueError("Scene image exceeds verification pixel budget")
            scenes.append({"scene": index, "artifact": str(artifact), "kind": "static"})
            frames.append({"scene": index, "path": str(artifact), "time_seconds": None})
        else:
            duration, stream = video_metadata(artifact)
            scenes.append({"scene": index, "artifact": str(artifact), "kind": "video",
                           "duration_seconds": duration, **stream})
            count = min(frames_per_video, stream["frame_count"])
            for frame_index in range(count):
                actual_index = round((stream["frame_count"] - 1) * frame_index / max(1, count - 1))
                timestamp = actual_index / stream["fps"]
                path = output / f"scene-{index:02d}-frame-{frame_index:02d}.png"
                subprocess.run(
                    ["ffmpeg", "-v", "error", "-nostdin", "-i", str(artifact), "-frames:v", "1",
                     "-vf", f"select=eq(n\\,{actual_index}),scale=512:-2", "-y", str(path)],
                    capture_output=True, timeout=30, check=True,
                )
                if not path.is_file() or not path.stat().st_size:
                    raise ValueError("Final video frame could not be decoded")
                frames.append({"scene": index, "path": str(path), "time_seconds": timestamp,
                               "frame_index": actual_index})
    return scenes, frames


def contact_sheet(frames, scenes, output):
    """A deterministic, timestamped judge preview; retain original videos separately."""
    from PIL import Image, ImageDraw, ImageOps

    if not frames or len(frames) > 48:
        raise ValueError("Expected between 1 and 48 evidence frames")
    columns, width, height = 4, 512, 332
    rows = math.ceil(len(frames) / columns)
    sheet = Image.new("RGB", (columns * width, rows * height + 40), "white")
    draw = ImageDraw.Draw(sheet)
    labels = [f"scene {s['scene']}: " + (f"{s['duration_seconds']:.3f}s" if s["kind"] == "video" else "static")
              for s in scenes]
    draw.text((8, 10), "; ".join(labels), fill="black")
    for index, frame in enumerate(frames):
        x, y = index % columns * width, index // columns * height + 40
        with Image.open(frame["path"]) as image:
            preview = ImageOps.contain(image.convert("RGB"), (width, height - 28))
            sheet.paste(preview, (x + (width - preview.width) // 2, y + 24))
        time = "static" if frame["time_seconds"] is None else f"t={frame['time_seconds']:.3f}s"
        draw.text((x + 8, y + 5), f"scene {frame['scene']} | {time}", fill="black")
    sheet.save(output)


def render(source, output, python, timeout=240):
    check_runtime(python)
    source, output = source.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    media = output / "media"
    try:
        result = subprocess.run(
            [python, str(Path(__file__).with_name("manim_entry.py")), "render", "-ql", "-a", "--disable_caching",
             "--format=mp4", "--media_dir", str(media), str(source)],
            cwd=source.parent, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"status": "execution_failed", "reason": "candidate_render_timeout"}
    (output / "render.log").write_text(result.stdout + "\n" + result.stderr)
    if result.returncode in (-9, -15, 137, 143):
        raise RuntimeUnavailable("Renderer was killed before a valid result")
    if result.returncode:
        return {"status": "execution_failed", "reason": "candidate_render_error", "exit_code": result.returncode}
    artifacts = final_artifacts(media)
    if not artifacts or len(artifacts) > 4:
        return {"status": "execution_failed", "reason": "expected_1_to_4_final_scenes"}
    try:
        scenes, frames = evidence_frames(artifacts, output)
        preview = output / "contact-sheet.png"
        contact_sheet(frames, scenes, preview)
    except (ValueError, KeyError, IndexError, OSError, subprocess.SubprocessError) as exc:
        # An undecodable artifact is not usable evidence for a quality measurement.
        raise RuntimeUnavailable("Final scene evidence extraction failed") from exc
    return {"status": "ok", "scenes": scenes, "frames": frames, "preview": str(preview),
            "temporal_evidence": "Up to 12 uniformly spaced decoded frames per video; timestamps and full durations retained"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--python", default=os.environ.get("MANIM_PYTHON", "/opt/manim/bin/python"))
    args = parser.parse_args()
    try:
        result = render(args.source, args.output, args.python)
    except RuntimeUnavailable as exc:
        result = {"status": "infrastructure_error", "reason": str(exc)}
    path = args.output.with_suffix(".result.json")
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "result": str(path)}))
    raise SystemExit(0 if result["status"] == "ok" else 20 if result["status"] == "execution_failed" else 30)


if __name__ == "__main__":
    main()
