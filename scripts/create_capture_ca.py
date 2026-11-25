"""Create a private capture CA once; only its certificate may enter sandboxes."""

import argparse
from pathlib import Path
import subprocess


def create(directory):
    directory = Path(directory)
    directory.mkdir(mode=0o700, parents=True, exist_ok=False)
    key = directory / "ca.key"
    with key.open("xb"):
        key.chmod(0o600)
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "ec",
            "-pkeyopt",
            "ec_paramgen_curve:P-256",
            "-nodes",
            "-keyout",
            str(key),
            "-out",
            str(directory / "ca.pem"),
            "-days",
            "30",
            "-sha256",
            "-subj",
            "/CN=MM Coding capture CA",
            "-addext",
            "basicConstraints=critical,CA:TRUE,pathlen:0",
            "-addext",
            "keyUsage=critical,digitalSignature,keyCertSign,cRLSign",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    create(args.output)
    print("Capture CA created; keep ca.key on the trusted Gym host.")
