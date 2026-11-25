"""Check native Gym OpenSandbox create/exec/upload/download/cleanup; no GPU work."""

import argparse
import asyncio
import json
import tempfile
from pathlib import Path

from mmrl.sandbox import RemoteSandbox


async def check(image):
    sandbox = RemoteSandbox(image, ttl=600)
    try:
        async with sandbox.session() as handle:
            await sandbox.prepare(handle)
            await sandbox.upload(
                handle, "/workspace/mmrl-probe.txt", "mm-coding native provider probe"
            )
            command = "python -c 'import json,platform; print(json.dumps(dict(architecture=platform.machine(),python=platform.python_version())))'"
            result = await sandbox.provider.exec(handle, command, timeout_s=30)
            if result.error_type or result.return_code != 0:
                raise RuntimeError("Native provider command failed")
            info = json.loads(result.stdout)
            with tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp) / "probe.txt"
                await sandbox.provider.download_file(
                    handle, "/workspace/mmrl-probe.txt", output
                )
                if output.read_text() != "mm-coding native provider probe":
                    raise RuntimeError("Native provider file round-trip failed")
            info["file_roundtrip"] = True
        info["cleanup"] = "completed"
        return info
    finally:
        await sandbox.provider.aclose()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        required=True,
        help="Digest-pinned CPU image with Python; no judge credentials are uploaded",
    )
    args = parser.parse_args()
    print(json.dumps(asyncio.run(check(args.image)), sort_keys=True))


if __name__ == "__main__":
    main()
