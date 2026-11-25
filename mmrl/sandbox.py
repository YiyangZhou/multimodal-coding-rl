import ipaddress
import os
import re
import shlex
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit

from .catalog import BENCHMARKS
from .reward import InfrastructureError

CAPTURE_CA_PATH = "/usr/local/share/ca-certificates/mmrl-capture.crt"


def validate_endpoint(value, label="OPENSANDBOX_DOMAIN"):
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path not in ("", "/")
    ):
        raise ValueError(f"{label} must be an explicit HTTPS origin without /v1")
    if parsed.port is not None and not 1 <= parsed.port <= 65535:
        raise ValueError(f"{label} has an invalid port")
    host = parsed.hostname.lower()
    if host in {"localhost", "0.0.0.0", "::"} or host.endswith(".localhost"):
        raise ValueError("OpenSandbox must be a remote endpoint")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    if address is not None and (address.is_loopback or address.is_unspecified):
        raise ValueError("OpenSandbox must be a remote endpoint")
    return value.rstrip("/")


def validate_image(image):
    if not re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", image):
        raise ValueError("Pin the sandbox OCI image by its complete SHA-256 digest")
    return image


def provider_config():
    domain = validate_endpoint(os.environ.get("OPENSANDBOX_DOMAIN", ""))
    key = os.environ.get("OPENSANDBOX_API_KEY")
    if not key:
        raise ValueError("OPENSANDBOX_API_KEY is missing")
    return {
        "opensandbox": {
            "connection": {
                "domain": domain,
                "api_key": key,
                "protocol": "https",
                "tls_verify": os.environ.get("OPENSANDBOX_TLS_VERIFY", "true").lower()
                != "false",
                "use_server_proxy": True,
                "transport_backend": "aiohttp",
                "request_timeout_s": 120,
            },
            "operations": {"command_retries": 0},
        }
    }


def image_pull_options(image):
    """Send registry credentials only to the matching image's control-plane pull."""
    host = os.environ.get("REGISTRY_HOST", "")
    username = os.environ.get("REGISTRY_USERNAME", "")
    password = os.environ.get("REGISTRY_PASSWORD", "")
    if not any((host, username, password)):
        return {}
    if not host or "/" in host or "://" in host:
        raise ValueError("REGISTRY_HOST must be an explicit registry host[:port]")
    if image.split("/", 1)[0] != host:
        return {}
    if not username or not password:
        raise ValueError("Image registry username and password are both required")
    return {"image_auth": {"username": username, "password": password}}


class RemoteSandbox:
    def __init__(self, image, provider=None, ttl=14400, *, memory_mib=16384, chartmimic_image=None):
        if isinstance(memory_mib, bool) or not isinstance(memory_mib, int) or memory_mib < 1024:
            raise ValueError("Sandbox memory must be an integer of at least 1024 MiB")
        self.image, self.ttl = validate_image(image), ttl
        self.memory_mib = memory_mib
        self.chartmimic_image = validate_image(chartmimic_image) if chartmimic_image else None
        if provider is None:
            from nemo_gym.sandbox.providers.registry import create_provider

            provider = create_provider(provider_config())
        self.provider = provider

    async def create(self):
        from nemo_gym.sandbox.providers.base import SandboxSpec

        # Gym's Hermes adapter uses non-streaming requests. Its 30-minute
        # default otherwise retries long generations before the three-hour
        # harness budget is exhausted. The outer rollout deadline still applies.
        env = {"HERMES_API_TIMEOUT": "10800"}
        if os.environ.get("MMRL_CAPTURE_CA_CERT"):
            env.update(
                NODE_EXTRA_CA_CERTS=CAPTURE_CA_PATH,
                SSL_CERT_FILE="/etc/ssl/certs/ca-certificates.crt",
                REQUESTS_CA_BUNDLE="/etc/ssl/certs/ca-certificates.crt",
            )
        return await self.provider.create(
            SandboxSpec(
                image=self.image,
                ttl_s=self.ttl,
                ready_timeout_s=600,
                workdir="/workspace",
                env=env,
                resources={"cpu": 4, "memory_mib": self.memory_mib, "disk_gib": 20},
                metadata={"mmrl/workload": "mm-coding"},
                provider_options=image_pull_options(self.image),
            )
        )

    @asynccontextmanager
    async def session(self):
        handle = await self.create()
        try:
            yield handle
        finally:
            await self.provider.close(handle)

    async def upload(self, handle, path, text):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "input"
            source.write_text(text)
            await self.provider.upload_file(handle, source, path)

    async def prepare(self, handle, original_source=""):
        result = await self.provider.exec(handle, "mkdir -p /workspace", timeout_s=30)
        if result.return_code != 0 or result.error_type:
            raise InfrastructureError("Sandbox workspace setup failed")
        # Login shells reset the image PATH and otherwise select system Python,
        # which lacks the locked rendering dependencies used by agent tools.
        await self.upload(
            handle,
            "/etc/profile.d/mmrl-python.sh",
            'export PATH="/opt/gym/bin:$PATH"\n',
        )
        if ca_file := os.environ.get("MMRL_CAPTURE_CA_CERT"):
            # Only the public CA is staged. Its signing key stays on the gateway host.
            certificate = Path(ca_file).read_text()
            if "PRIVATE KEY" in certificate or not certificate.startswith(
                "-----BEGIN CERTIFICATE-----"
            ):
                raise ValueError(
                    "Capture CA file must contain only public certificates"
                )
            await self.upload(handle, CAPTURE_CA_PATH, certificate)
            result = await self.provider.exec(
                handle, "update-ca-certificates", timeout_s=60
            )
            if result.error_type or result.return_code != 0:
                raise InfrastructureError("Sandbox capture CA installation failed")
        if original_source:
            await self.upload(handle, "/workspace/original.html", original_source)

    async def read_artifact(self, handle, name):
        # Only catalog filenames; fail absent/oversized artifacts as model output errors.
        if name not in {v.artifact for v in BENCHMARKS.values()}:
            raise ValueError("Unknown artifact name")
        path = "/workspace/" + name
        quoted = shlex.quote(path)
        command = (
            f"test ! -L {quoted} || exit 42; "
            f"test -f {quoted} || exit 41; "
            f"size=$(wc -c < {quoted}) || exit 44; "
            'test "$size" -le 2097152 || exit 43'
        )
        result = await self.provider.exec(handle, command, timeout_s=30)
        if result.error_type:
            raise InfrastructureError("Sandbox artifact read failed")
        if result.return_code != 0:
            reason = {
                41: "artifact_missing_or_not_regular",
                42: "artifact_symlink",
                43: "artifact_oversize",
                44: "artifact_size_probe_failed",
            }.get(result.return_code, "artifact_probe_failed")
            raise ValueError(reason)
        with tempfile.TemporaryDirectory() as tmp:
            local = Path(tmp) / name
            await self.provider.download_file(handle, path, local)
            return local.read_text()

    async def render(self, benchmark, candidate):
        if benchmark == "3dcodebench":
            evidence = await self.render_blender(candidate)
            return evidence.png if evidence is not None else None
        if benchmark == "manimbench_v1":
            evidence = await self.render_manim(candidate)
            return evidence.png if evidence is not None else None
        # Fresh verification box: the agent cannot substitute its own preview PNG.
        async with self.session() as handle:
            await self.prepare(handle)
            if benchmark == "image2struct_latex":
                # Keep the body preamble paired with this verifier release,
                # including when reusing an older pinned sandbox image.
                await self.upload(handle, "/opt/mmrl/render.py",
                                  Path(__file__).with_name("render.py").read_text())
            check = await self.provider.exec(
                handle, "python /opt/mmrl/render.py --check", timeout_s=60
            )
            if check.error_type or check.return_code != 0:
                raise InfrastructureError("Sandbox renderer runtime is incomplete")
            filename = (
                "answer.html"
                if benchmark == "webmmu_code_edit"
                else BENCHMARKS[benchmark].artifact
            )
            await self.upload(handle, "/workspace/" + filename, candidate)
            result = await self.provider.exec(
                handle,
                f"python /opt/mmrl/render.py {benchmark} /workspace/{filename} /workspace/result.png",
                timeout_s=180,
            )
            if result.error_type or result.return_code in {137, 143}:
                raise InfrastructureError("Rendering infrastructure failed")
            if result.return_code != 0:
                return None
            with tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp) / "result.png"
                await self.provider.download_file(
                    handle, "/workspace/result.png", output
                )
                if output.stat().st_size > 32 * 1024 * 1024:
                    raise ValueError("Rendered artifact exceeds 32 MiB")
                return output.read_bytes()

    async def render_manim(self, candidate, *, media_directory=None):
        from .manim_sandbox import render_manim

        return await render_manim(self, candidate, media_directory=media_directory)

    async def score_vision(self, task, candidate, judge, *, audit=None):
        from .vision_sandbox import score_vision

        return await score_vision(self, task, candidate, judge, audit)

    async def render_blender(self, candidate, *, media_directory=None):
        from .blender_sandbox import render_blender

        return await render_blender(self, candidate, media_directory=media_directory)

    async def render_chartmimic(self, candidate):
        from .chartmimic_sandbox import render_chartmimic

        if not self.chartmimic_image:
            raise InfrastructureError("Configure a pinned ChartMimic Python 3.9 renderer image")
        renderer = RemoteSandbox(self.chartmimic_image, provider=self.provider,
                                 ttl=600, memory_mib=self.memory_mib)
        return await render_chartmimic(renderer, candidate)
