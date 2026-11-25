"""Canonical four-view Blender verification, orchestrated from the trusted host."""

from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import shlex
import tempfile

from .render_transport import download_bounded
from .reward import InfrastructureError

VIEWS = (("Image_005.png", 45), ("Image_015.png", 135),
         ("Image_025.png", 225), ("Image_035.png", 315))
RENDERER_SHA256 = "0b026a835e385338b396604ccb48d2555ff3ef74786cb4ba80ff83f78fbc1934"


@dataclass(frozen=True)
class ObjectEvidence:
    png: bytes
    metadata: dict


def classify_receipt(receipt):
    if not isinstance(receipt, dict):
        raise InfrastructureError("Invalid Blender verification receipt")
    status = receipt.get("status")
    # Upstream exits 0 even for ERR_EXEC; the log is mandatory.
    if status in {"ERR_EXEC", "ERR_NO_MESH"}:
        return False
    meshes = receipt.get("n_meshes")
    if status != "OK" or receipt.get("n_views_rendered") != 4 or type(meshes) is not int or meshes < 1:
        raise InfrastructureError("Blender did not produce four verified object views")
    return True


def contact_sheet(directory):
    from PIL import Image, ImageDraw

    sheet = Image.new("RGB", (1024, 1080), "white")
    draw = ImageDraw.Draw(sheet)
    hashes = {}
    for index, (name, angle) in enumerate(VIEWS):
        path = directory / name
        try:
            with Image.open(path) as im:
                if im.format != "PNG" or im.size != (512, 512):
                    raise ValueError("Noncanonical render dimensions")
                im.load()
                x, y = index % 2 * 512, index // 2 * 540
                draw.text((x + 8, y + 8), f"Azimuth {angle} degrees", fill="black")
                background = Image.new("RGBA", im.size, (3, 3, 5, 255))
                background.alpha_composite(im.convert("RGBA"))
                sheet.paste(background.convert("RGB"), (x, y + 28))
        except (OSError, ValueError) as exc:
            raise InfrastructureError("Blender view is missing or undecodable") from exc
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    buffer = io.BytesIO()
    sheet.save(buffer, format="PNG")
    return buffer.getvalue(), hashes


async def render_blender(sandbox, candidate, *, media_directory=None):
    async with sandbox.session() as handle:
        await sandbox.prepare(handle)
        preflight = (
            "import hashlib,subprocess; from pathlib import Path; "
            "p=Path('/opt/3dcodebench/render.py'); "
            f"assert hashlib.sha256(p.read_bytes()).hexdigest()=={RENDERER_SHA256!r}; "
            "r=subprocess.run(['/opt/blender/blender','--background','--version'], "
            "capture_output=True,text=True,timeout=30,check=True); "
            "assert r.stdout.startswith('Blender 5.0.0')"
        )
        check = await sandbox.provider.exec(handle, "python -c " + shlex.quote(preflight), timeout_s=60)
        if check.error_type or check.return_code:
            raise InfrastructureError("Pinned Blender runtime unavailable")
        # Blender embeds its own Python; host/sandbox Python packages do not apply.
        dependency_command = (
            "for blender_python in /opt/blender/*/python/bin/python3*; do "
            'if [ -x "$blender_python" ]; then '
            'uv pip install --python "$blender_python" '
            "numpy==1.26.4 scipy==1.16.2 shapely==2.1.2 trimesh==4.8.3 scikit-image==0.25.2 "
            '&& "$blender_python" -c '
            + shlex.quote("import numpy, scipy, shapely, trimesh, skimage; "
                          "assert (numpy.__version__, scipy.__version__, shapely.__version__, "
                          "trimesh.__version__, skimage.__version__) == "
                          "('1.26.4','1.16.2','2.1.2','4.8.3','0.25.2')")
            + "; exit $?; fi; done; exit 1"
        )
        dependencies = await sandbox.provider.exec(handle, dependency_command, timeout_s=300)
        if dependencies.error_type or dependencies.return_code:
            raise InfrastructureError("Pinned Blender Python dependencies unavailable")
        await sandbox.upload(handle, "/workspace/answer.py", candidate)
        entry = "/opt/3dcodebench/render.py"
        if media_directory is not None:
            entry = "/workspace/blender_media_entry.py"
            await sandbox.upload(handle, entry, Path(__file__).with_name("blender_media_entry.py").read_text())
        result = await sandbox.provider.exec(handle,
            "/opt/blender/blender --background --factory-startup --threads 4 "
            f"--python {entry} -- --blender-render "
            "--script /workspace/answer.py --output-dir /workspace/views "
            "--samples 64 --resolution 512 --engine CYCLES", timeout_s=300)
        if result.error_type or result.return_code:
            raise InfrastructureError("Blender verification process failed without a usable receipt")
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            await download_bounded(sandbox, handle, "/workspace/views/render_log.json",
                                   directory / "render_log.json", 1024 * 1024)
            try:
                receipt = json.loads((directory / "render_log.json").read_text())
            except ValueError as exc:
                raise InfrastructureError("Unreadable Blender verification receipt") from exc
            if not classify_receipt(receipt):
                return None
            for name, _ in VIEWS:
                await download_bounded(sandbox, handle, "/workspace/views/" + name,
                                       directory / name, 8 * 1024 * 1024)
            png, hashes = contact_sheet(directory)
            media = None
            if media_directory is not None:
                from .media_capture import capture_object
                media = await capture_object(sandbox, handle, Path(media_directory))
            return ObjectEvidence(png, {"full_media_capture": media, "renderer_sha256": RENDERER_SHA256,
                "resolution": 512, "samples": 64, "engine": "CYCLES", "views_sha256": hashes,
                "azimuth_degrees": [angle for _, angle in VIEWS], "render_log": receipt})
