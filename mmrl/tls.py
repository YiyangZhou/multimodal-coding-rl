"""Private CA issuance and a Gym-owned capture gateway lifecycle."""

from contextlib import asynccontextmanager
import ipaddress
import os
from pathlib import Path
import ssl
import secrets
import subprocess
import tempfile


def openssl(*args):
    return subprocess.check_output(["openssl", *map(str, args)], stderr=subprocess.PIPE)


def issue_certificate(ca_file, ca_key_file, host, directory):
    address = ipaddress.ip_address(host)
    if not address.is_private or address.is_unspecified or address.is_multicast:
        raise ValueError("Capture gateway requires a specific private IP")
    ca_key_file = Path(ca_key_file)
    if ca_key_file.is_symlink() or ca_key_file.stat().st_mode & 0o077:
        raise ValueError("Capture CA key must be a private regular file")
    if openssl("x509", "-in", ca_file, "-pubkey", "-noout") != openssl(
        "pkey", "-in", ca_key_file, "-pubout"
    ):
        raise ValueError("Capture CA certificate and key do not match")
    if b"CA:TRUE" not in openssl(
        "x509", "-in", ca_file, "-noout", "-ext", "basicConstraints"
    ):
        raise ValueError("Capture certificate is not a CA")
    openssl("verify", "-CAfile", ca_file, ca_file)
    directory = Path(directory)
    key_path, cert_path = directory / "server.key", directory / "server.pem"
    with key_path.open("xb"):
        key_path.chmod(0o600)
    csr = directory / "server.csr"
    extensions = directory / "server.ext"
    extensions.write_text(
        f"subjectAltName=IP:{host}\nbasicConstraints=critical,CA:FALSE\nextendedKeyUsage=serverAuth\n"
        "subjectKeyIdentifier=hash\nauthorityKeyIdentifier=keyid,issuer\n"
    )
    openssl(
        "req",
        "-new",
        "-newkey",
        "ec",
        "-pkeyopt",
        "ec_paramgen_curve:P-256",
        "-nodes",
        "-keyout",
        key_path,
        "-out",
        csr,
        "-subj",
        f"/CN={host}",
    )
    openssl(
        "x509",
        "-req",
        "-in",
        csr,
        "-CA",
        ca_file,
        "-CAkey",
        ca_key_file,
        "-set_serial",
        "0x" + secrets.token_hex(16),
        "-days",
        "7",
        "-sha256",
        "-extfile",
        extensions,
        "-out",
        cert_path,
    )
    return cert_path, key_path


@asynccontextmanager
async def managed_gateway(port, host, global_config, *, response_observer=None, tool_upstream=None):
    if port is None:
        yield
        return
    from aiohttp import web
    from nemo_gym.global_config import get_first_server_config_dict
    from .gateway import create_app

    policy = get_first_server_config_dict(global_config, "policy_model")
    upstream = f"http://{policy.host}:{policy.port}"
    with tempfile.TemporaryDirectory(prefix="mmrl-capture-tls-") as temporary:
        cert, key = issue_certificate(
            os.environ["MMRL_CAPTURE_CA_CERT"],
            os.environ["MMRL_CAPTURE_CA_KEY"],
            host,
            temporary,
        )
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.load_cert_chain(cert, key)
        runner = web.AppRunner(
            create_app(upstream, os.environ["MMRL_CAPTURE_GATEWAY_TOKEN"],
                       response_observer=response_observer, tool_upstream=tool_upstream),
            access_log=None,
        )
        try:
            await runner.setup()
            await web.TCPSite(runner, host, port, ssl_context=context).start()
            yield
        finally:
            await runner.cleanup()
