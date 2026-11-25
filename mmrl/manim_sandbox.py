"""Trusted-side transport for final Manim evidence from a fresh CPU sandbox."""

from dataclasses import dataclass
import io
import json
from pathlib import Path
import tempfile

from .reward import InfrastructureError
from .render_transport import download_bounded


@dataclass(frozen=True)
class AnimationEvidence:
    png: bytes
    metadata: dict


async def render_manim(sandbox, candidate, *, media_directory=None):
    # No judge credentials, private reference code, or agent-produced previews are
    # present in a candidate box. The reference gets a separate fresh box.
    async with sandbox.session() as handle:
        await sandbox.prepare(handle)
        await sandbox.upload(handle, "/opt/mmrl/mmrl/manim_entry.py",
                             Path(__file__).with_name("manim_entry.py").read_text())
        await sandbox.upload(handle, "/workspace/answer.py", candidate)
        result = await sandbox.provider.exec(
            handle,
            "python /opt/mmrl/mmrl/manim_render.py /workspace/answer.py /workspace/evidence",
            timeout_s=360,
        )
        if result.error_type or result.return_code not in {0, 20}:
            raise InfrastructureError("Manim renderer runtime or transport failed")
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            await download_bounded(sandbox, handle, "/workspace/evidence.result.json",
                                   directory / "result.json", 1024 * 1024)
            try:
                metadata = json.loads((directory / "result.json").read_text())
                status = metadata["status"]
                if status == "execution_failed" and result.return_code == 20:
                    return None
                if status != "ok" or result.return_code != 0:
                    raise ValueError("Exit status and receipt disagree")
                if metadata["preview"] != "/workspace/evidence/contact-sheet.png":
                    raise ValueError("Unexpected evidence path")
                if not 1 <= len(metadata["scenes"]) <= 4 or not 1 <= len(metadata["frames"]) <= 48:
                    raise ValueError("Missing final scene evidence")
            except (KeyError, TypeError, ValueError) as exc:
                raise InfrastructureError("Invalid Manim verification receipt") from exc
            await download_bounded(sandbox, handle, "/workspace/evidence/contact-sheet.png",
                                   directory / "preview.png", 32 * 1024 * 1024)
            png = (directory / "preview.png").read_bytes()
            try:
                from PIL import Image

                with Image.open(io.BytesIO(png)) as image:
                    if image.format != "PNG" or image.width * image.height > 32_000_000:
                        raise ValueError("Invalid animation preview image")
                    image.verify()
            except (OSError, ValueError) as exc:
                raise InfrastructureError("Manim animation preview is undecodable") from exc
            if media_directory is not None:
                from .media_capture import capture_animation
                metadata["full_media_capture"] = await capture_animation(
                    sandbox, handle, metadata, Path(media_directory)
                )
            return AnimationEvidence(png, metadata)
