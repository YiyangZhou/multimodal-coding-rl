"""Transport for a dedicated, version-pinned ChartMimic renderer image."""
import io
import json
from pathlib import Path
import tempfile

from .render_transport import download_bounded
from .reward import InfrastructureError


async def render_chartmimic(sandbox, candidate):
    async with sandbox.session() as handle:
        await sandbox.prepare(handle)
        check = await sandbox.provider.exec(handle, "python /opt/mmrl/chartmimic_render.py --check", timeout_s=60)
        if check.error_type or check.return_code:
            raise InfrastructureError("Pinned ChartMimic renderer is unavailable")
        await sandbox.upload(handle, "/workspace/answer.py", candidate)
        result = await sandbox.provider.exec(handle,
            "python /opt/mmrl/chartmimic_render.py /workspace/answer.py", timeout_s=210)
        if result.error_type or result.return_code not in {0, 20}:
            raise InfrastructureError("ChartMimic renderer or transport failed")
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            await download_bounded(sandbox, handle, "/workspace/chart.result.json", directory / "result.json", 4096)
            try:
                receipt = json.loads((directory / "result.json").read_text())
                if receipt["exit_code"] != result.return_code:
                    raise ValueError("Exit status mismatch")
                if result.return_code == 20 and receipt["status"] in {
                    "candidate_timeout", "candidate_execution_failed", "invalid_pdf_size",
                    "expected_single_page", "invalid_page_size", "invalid_pdf",
                }:
                    return None
                if result.return_code != 0 or receipt["status"] != "ok":
                    raise ValueError("Invalid rendering result")
            except (TypeError, ValueError, KeyError) as exc:
                raise InfrastructureError("Invalid ChartMimic rendering receipt") from exc
            await download_bounded(sandbox, handle, "/workspace/chart.png", directory / "chart.png", 32 * 1024 * 1024)
            png = (directory / "chart.png").read_bytes()
            from PIL import Image
            try:
                with Image.open(io.BytesIO(png)) as image:
                    if image.format != "PNG" or image.width * image.height > 32_000_000:
                        raise ValueError("Invalid chart preview")
                    image.verify()
            except (ValueError, OSError) as exc:
                raise InfrastructureError("Invalid chart preview") from exc
            return png
