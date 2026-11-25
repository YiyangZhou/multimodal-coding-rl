"""Install pinned public renderer artifacts inside the amd64 CPU image builder."""

import hashlib
import json
from pathlib import Path
import platform
import tarfile
import urllib.request


def fetch(url, sha256, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=180) as response, destination.open("xb") as out:
        while block := response.read(1024 * 1024):
            out.write(block)
            digest.update(block)
    if digest.hexdigest() != sha256:
        raise ValueError("Public renderer artifact checksum mismatch")


def main():
    if platform.machine() != "x86_64":
        raise ValueError("This CPU sandbox renderer image is pinned for linux/amd64")
    locks = Path("/opt/locks")
    blender = json.loads((locks / "blender.lock.json").read_text())
    archive = Path("/tmp/blender.tar.xz")
    fetch(blender["url"], blender["sha256"], archive)
    with tarfile.open(archive) as tar:
        tar.extractall("/opt", filter="data")
    Path("/opt/blender").symlink_to("/opt/blender-5.0.0-linux-x64", target_is_directory=True)
    archive.unlink()
    url = ("https://raw.githubusercontent.com/gaoypeng/3dcodebench/"
           + blender["upstream_renderer_commit"] + "/" + blender["upstream_renderer_path"])
    fetch(url, blender["upstream_renderer_sha256"], Path("/opt/3dcodebench/render.py"))
    license_url = ("https://raw.githubusercontent.com/gaoypeng/3dcodebench/"
                   + blender["upstream_renderer_commit"] + "/" + blender["upstream_license_path"])
    fetch(license_url, blender["upstream_license_sha256"], Path("/opt/3dcodebench/LICENSE"))
    assets = json.loads((locks / "manim.assets.lock.json").read_text())
    fetch(assets["url"], assets["sha256"], Path(assets["container_path"]))
    fetch(assets["license_url"], assets["license_sha256"], Path("/opt/mmrl/assets/us-atlas-LICENSE"))
    Path("/opt/mmrl/assets/manim.assets.lock.json").write_text(json.dumps(assets))


if __name__ == "__main__":
    main()
