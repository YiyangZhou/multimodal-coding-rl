import hashlib
import json
from unittest.mock import patch

import pytest
import requests

from mmrl.manim_assets import install_asset_cache


def test_only_exact_public_get_is_cached(tmp_path):
    data = b'{"type":"Topology","arcs":[]}'
    asset = tmp_path / "map.json"
    asset.write_bytes(data)
    lock = tmp_path / "lock.json"
    url = "https://cdn.jsdelivr.net/npm/us-atlas@3/nation-10m.json"
    lock.write_text(json.dumps({"container_path": str(asset), "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(), "url": url.replace("@3/", "@3.0.1/"),
        "original_url": url}))
    with patch("requests.sessions.Session.send") as upstream:
        install_asset_cache(lock)
        response = requests.get(url)
        assert response.json()["type"] == "Topology"
        assert response.content == data
        upstream.assert_not_called()
        requests.get(url + "?different=request")
        assert upstream.call_count == 1
        requests.post(url, data="not a read")
        assert upstream.call_count == 2


def test_corrupt_public_asset_fails_before_code_execution(tmp_path):
    asset = tmp_path / "map.json"
    asset.write_bytes(b"wrong")
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({"container_path": str(asset), "bytes": 5, "sha256": "a" * 64}))
    with pytest.raises(ValueError, match="pinned snapshot"):
        install_asset_cache(lock)
