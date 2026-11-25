"""Sandbox-only startup of a single-turn web project; no host execution."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time
import urllib.error
import urllib.request

ROOT = Path("/workspace")
NETWORK_FAILURE = re.compile(
    r"EAI_AGAIN|ENETUNREACH|ECONNRESET|ETIMEDOUT|Temporary failure in name resolution|"
    r"Could not resolve host|Connection timed out|CERTIFICATE_VERIFY_FAILED|"
    r"503 Service Unavailable|502 Bad Gateway|429 Too Many Requests", re.I)


def receipt(status, reason, **details):
    (ROOT / ".mmrl/deployment.json").write_text(json.dumps({"status": status, "reason": reason, **details}))
    return {"ready": 0, "execution_failed": 20, "infrastructure_error": 30}[status]


def check_environment():
    for executable in ("bash", "python", "node", "npm"):
        if not shutil.which(executable):
            raise RuntimeError("Missing web runtime")
    expected = json.loads((ROOT / ".mmrl/resources.json").read_text())
    for item in expected:
        path = ROOT / item["path"]
        if path.is_symlink() or not path.is_file() or path.stat().st_size != item["bytes"]:
            raise RuntimeError("Resource staging failure")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != item["sha256"]:
            raise RuntimeError("Resource staging checksum mismatch")


def run(deadline_seconds=300):
    try:
        check_environment()
    except (OSError, ValueError, KeyError, TypeError, RuntimeError):
        return receipt("infrastructure_error", "web_runtime_or_dataset_staging_unavailable")
    log_path = ROOT / ".mmrl/deployment.log"
    process = None
    ready = False
    try:
        with log_path.open("wb") as log:
            process = subprocess.Popen(["bash", str(ROOT / "start.sh")], cwd=ROOT,
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                       start_new_session=True)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        started = time.monotonic()
        while time.monotonic() - started < deadline_seconds:
            if log_path.stat().st_size > 4 * 1024 * 1024:
                return receipt("infrastructure_error", "deployment_diagnostics_exceeded_budget")
            try:
                with opener.open("http://127.0.0.1:3000", timeout=2) as response:
                    if 200 <= response.status < 400:
                        ready = True
                        return receipt("ready", "application_listening", http_status=response.status)
            except (urllib.error.URLError, TimeoutError, OSError):
                pass
            code = process.poll()
            if code is not None and (code != 0 or time.monotonic() - started > 5):
                logs = log_path.read_text(errors="replace")
                if code < 0 or NETWORK_FAILURE.search(logs):
                    return receipt("infrastructure_error", "deployment_network_or_process_interruption", exit_code=code)
                return receipt("execution_failed", "start_script_exited_without_application", exit_code=code)
            time.sleep(1)
        # A slow dependency install is not a measured model-quality failure.
        return receipt("infrastructure_error", "deployment_deadline_without_measurement")
    except OSError:
        return receipt("infrastructure_error", "deployment_process_unavailable")
    finally:
        if process is not None and not ready:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass


if __name__ == "__main__":
    raise SystemExit(run())
