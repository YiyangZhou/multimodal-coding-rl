"""Preserve full final media before the verification sandbox is destroyed.

Optional observability: failure is reported separately and never changes reward.
"""
import hashlib
from pathlib import PurePosixPath
from .render_transport import download_bounded

async def capture_animation(sandbox, handle, metadata, destination):
    report = {'complete': False, 'files': []}
    try:
        destination.mkdir(parents=True, exist_ok=False)
        scenes = metadata['scenes']
        if not 1 <= len(scenes) <= 4:
            raise ValueError('Invalid scene count')
        for index, scene in enumerate(scenes):
            remote = PurePosixPath(scene['artifact'])
            root = PurePosixPath('/workspace/evidence/media')
            if '..' in remote.parts or not remote.is_relative_to(root):
                raise ValueError('Unexpected media location')
            suffix = '.mp4' if scene['kind'] == 'video' else '.png' if scene['kind'] == 'static' else None
            if suffix is None or remote.suffix != suffix or 'partial_movie_files' in remote.parts:
                raise ValueError('Not a complete final media artifact')
            local = destination / f'scene-{index:02d}{suffix}'
            await download_bounded(sandbox, handle, str(remote), local, 128 * 1024 * 1024)
            report['files'].append({'scene': index, 'kind': scene['kind'], 'path': str(local),
                                    'bytes': local.stat().st_size,
                                    'sha256': hashlib.sha256(local.read_bytes()).hexdigest()})
        report['complete'] = True
    except Exception as exc:
        report['error_type'] = type(exc).__name__
    return report


async def capture_object(sandbox, handle, destination):
    import json
    report = {"complete": False, "files": []}
    try:
        destination.mkdir(parents=True, exist_ok=False)
        receipt = destination / "media-export.json"
        await download_bounded(sandbox, handle, "/workspace/views/media-export.json", receipt, 65536)
        status = json.loads(receipt.read_text())
        if not status.get("complete") or not status.get("same_render_process"):
            raise ValueError("Same-scene export unavailable")
        local = destination / "model.glb"
        await download_bounded(sandbox, handle, "/workspace/views/model.glb", local, 128 * 1024 * 1024)
        with local.open("rb") as stream:
            if stream.read(4) != b"glTF":
                raise ValueError("Invalid GLB header")
        report.update(complete=True, same_render_process=True,
                      files=[{"path": str(local), "bytes": local.stat().st_size,
                              "sha256": hashlib.sha256(local.read_bytes()).hexdigest()}])
    except Exception as exc:
        report["error_type"] = type(exc).__name__
    return report
