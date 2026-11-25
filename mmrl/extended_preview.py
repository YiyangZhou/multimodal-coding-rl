"""Sandbox-only previews using the same pinned renderers as final verification.

No reference artifacts, grader keys, or private scores are consumed here.
Installed into each private task sandbox before the harness starts.
"""
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile



VIEWS = (("Image_005.png", 45), ("Image_015.png", 135),
         ("Image_025.png", 225), ("Image_035.png", 315))

RENDERER_SHA256 = "0b026a835e385338b396604ccb48d2555ff3ef74786cb4ba80ff83f78fbc1934"

def classify_receipt(receipt):
    if not isinstance(receipt, dict):
        raise RuntimeError("Invalid Blender verification receipt")
    status = receipt.get("status")
    # Upstream exits 0 even for ERR_EXEC; the log is mandatory.
    if status in {"ERR_EXEC", "ERR_NO_MESH"}:
        return False
    meshes = receipt.get("n_meshes")
    if status != "OK" or receipt.get("n_views_rendered") != 4 or type(meshes) is not int or meshes < 1:
        raise RuntimeError("Blender did not produce four verified object views")
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
            raise RuntimeError("Blender view is missing or undecodable") from exc
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    buffer = io.BytesIO()
    sheet.save(buffer, format="PNG")
    return buffer.getvalue(), hashes

def render(benchmark, source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    # Every invocation uses a fresh output directory: never return stale evidence.
    with tempfile.TemporaryDirectory(prefix='mmrl-preview-', dir=source.parent) as tmp:
        root = Path(tmp)
        if benchmark == 'manimbench_v1':
            entry = Path('/opt/mmrl/mmrl/manim_render.py')
            evidence = root / 'evidence'
            process = subprocess.run([sys.executable, str(entry), str(source), str(evidence)],
                                     cwd=source.parent, timeout=360)
            receipt = json.loads(evidence.with_suffix('.result.json').read_text())
            if process.returncode == 20 and receipt.get('status') == 'execution_failed':
                raise ValueError('Candidate Manim execution failed')
            if process.returncode or receipt.get('status') != 'ok':
                raise RuntimeError('Manim preview runtime failed')
            preview = evidence / 'contact-sheet.png'
            if Path(receipt['preview']).resolve() != preview.resolve():
                raise RuntimeError('Unexpected Manim evidence path')
            shutil.copyfile(preview, output)
        elif benchmark == '3dcodebench':
            entry = Path('/opt/3dcodebench/render.py')
            if hashlib.sha256(entry.read_bytes()).hexdigest() != RENDERER_SHA256:
                raise RuntimeError('Blender renderer source mismatch')
            views = root / 'views'
            subprocess.run(['/opt/blender/blender', '--background', '--factory-startup',
                            '--threads', '4', '--python', str(entry), '--', '--blender-render',
                            '--script', str(source), '--output-dir', str(views),
                            '--samples', '64', '--resolution', '512', '--engine', 'CYCLES'],
                           cwd=source.parent, timeout=300, check=True)
            receipt = json.loads((views / 'render_log.json').read_text())
            if not classify_receipt(receipt):
                raise ValueError('Candidate Blender execution failed')
            png, _ = contact_sheet(views)
            output.write_bytes(png)
        else:
            raise ValueError('Unsupported extended preview benchmark')
