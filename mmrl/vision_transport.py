"""Host-side browser control for one existing verification sandbox."""
import io
import json
from pathlib import Path
import tempfile

from .render_transport import download_bounded
from .reward import InfrastructureError
from .vision_browser import validate_action


class VisionBrowserSession:
    def __init__(self, sandbox, handle):
        self.sandbox, self.handle = sandbox, handle
        self.viewport = None

    async def prepare(self):
        result = await self.sandbox.provider.exec(self.handle, "mkdir -p /workspace/.mmrl", timeout_s=30)
        if result.error_type or result.return_code:
            raise InfrastructureError("Browser workspace initialization failed")
        # Only verifier code enters this box; model API keys stay on the host.
        await self.sandbox.upload(self.handle, "/workspace/.mmrl/browser.py",
                                  Path(__file__).with_name("vision_browser.py").read_text())

    async def _invoke(self, stop=False):
        result = await self.sandbox.provider.exec(self.handle,
            "python /workspace/.mmrl/browser.py " + ("--stop" if stop else "--request"), timeout_s=90)
        if result.error_type or result.return_code:
            raise InfrastructureError("Browser command failed at the runtime/transport boundary")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "reply.json"
            await download_bounded(self.sandbox, self.handle, "/workspace/.mmrl/reply.json", path, 262144)
            try:
                value = json.loads(path.read_text())
                if value["status"] not in {"ok", "app_failure"}:
                    raise ValueError("Unusable browser status")
            except (ValueError, TypeError, KeyError) as exc:
                raise InfrastructureError("Invalid browser reply") from exc
            return value

    async def command(self, action):
        validate_action(action)
        if action["op"] == "start":
            self.viewport = (action["width"], action["height"])
        await self.sandbox.upload(self.handle, "/workspace/.mmrl/request.json", json.dumps(action))
        return await self._invoke()

    async def screenshot(self, *, fullpage):
        value = await self.command({"op": "screenshot", "fullpage": fullpage})
        if value != {"status": "ok", "screenshot": "/workspace/.mmrl/screenshot.png"}:
            raise InfrastructureError("Browser did not produce the requested visual checkpoint")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "screenshot.png"
            await download_bounded(self.sandbox, self.handle, "/workspace/.mmrl/screenshot.png", path, 32 * 1024 * 1024)
            png = path.read_bytes()
            try:
                from PIL import Image
                with Image.open(io.BytesIO(png)) as image:
                    if image.format != "PNG" or image.width * image.height > 32_000_000:
                        raise ValueError("Invalid browser screenshot")
                    if not fullpage and (self.viewport is None or image.size != self.viewport):
                        raise ValueError("Browser screenshot differs from the workflow viewport")
                    image.verify()
            except (ValueError, OSError) as exc:
                raise InfrastructureError("Browser screenshot is not valid evidence") from exc
            return png

    async def stop(self):
        value = await self._invoke(stop=True)
        if value != {"status": "ok"}:
            raise InfrastructureError("Browser workflow cleanup failed")
