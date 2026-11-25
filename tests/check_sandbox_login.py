"""Check agent-tool Python resolution in a real, temporary OpenSandbox."""
import argparse
import asyncio
import json
import shlex

from mmrl.sandbox import RemoteSandbox


async def check(image: str) -> dict:
    sandbox = RemoteSandbox(image, ttl=600)
    code = (
        "import sys,json,importlib.util; print(json.dumps({'executable':sys.executable,"
        "'modules':{k:importlib.util.find_spec(k) is not None "
        "for k in ['PIL','numpy','matplotlib','cairosvg']}}))"
    )
    command = "python3 -c " + shlex.quote(code)
    results = {}
    try:
        async with sandbox.session() as handle:
            await sandbox.prepare(handle)
            for name, shell in (("default", None), ("nonlogin", "bash -c"), ("login", "bash -lc")):
                result = await sandbox.provider.exec(
                    handle,
                    command if shell is None else shell + " " + shlex.quote(command),
                    timeout_s=30,
                )
                if result.error_type or result.return_code != 0:
                    raise RuntimeError(f"Python probe failed for {name}")
                row = json.loads(result.stdout)
                assert row["executable"].startswith("/opt/gym/bin/"), row
                assert all(row["modules"].values()), row
                results[name] = row
        return {"passed": True, "cleanup": True, "results": results}
    finally:
        await sandbox.provider.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(check(args.image)), indent=2))
