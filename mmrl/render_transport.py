"""Bounded evidence transfer from a disposable verification sandbox."""

import shlex

from .reward import InfrastructureError


async def download_bounded(sandbox, handle, remote, local, limit):
    path = shlex.quote(remote)
    check = await sandbox.provider.exec(
        handle, f"test -f {path} && test ! -L {path} && test $(wc -c < {path}) -le {limit}",
        timeout_s=30,
    )
    if check.error_type or check.return_code:
        raise InfrastructureError("Verification evidence is missing or oversized")
    await sandbox.provider.download_file(handle, remote, local)
    if not 0 < local.stat().st_size <= limit:
        raise InfrastructureError("Invalid verification evidence size")
