"""Serve the official Manim map dependency from a hash-verified local snapshot."""

import hashlib
import json
from pathlib import Path


def install_asset_cache(lock_path=Path("/opt/mmrl/assets/manim.assets.lock.json")):
    import requests

    lock = json.loads(lock_path.read_text())
    data = Path(lock["container_path"]).read_bytes()
    if len(data) != lock["bytes"] or hashlib.sha256(data).hexdigest() != lock["sha256"]:
        raise ValueError("Manim public asset does not match its pinned snapshot")
    original = requests.sessions.Session.send
    urls = {lock["url"], lock["original_url"]}

    def send(session, request, **kwargs):
        if request.method != "GET" or request.url not in urls:
            return original(session, request, **kwargs)
        response = requests.Response()
        response.status_code = 200
        response.url = request.url
        response.request = request
        response.headers["Content-Type"] = "application/json"
        response.headers["Content-Length"] = str(len(data))
        response._content = data
        response.encoding = "utf-8"
        return response

    requests.sessions.Session.send = send
