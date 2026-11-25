"""Install pinned preview entrypoints into a new private task sandbox."""
from pathlib import Path

async def install_extended_preview(sandbox, handle, benchmark):
    if benchmark not in {'manimbench_v1', '3dcodebench'}:
        return
    root = Path(__file__).resolve().parent
    for name, destination in (
        ('extended_preview.py', '/opt/mmrl/mmrl/extended_preview.py'),
        ('render.py', '/opt/mmrl/render.py'),
    ):
        await sandbox.upload(handle, destination, (root / name).read_text())
