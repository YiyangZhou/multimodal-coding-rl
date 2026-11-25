"""Python 3.9 renderer entrypoint, executed only inside a disposable sandbox.

This is the strict training execution check, not upstream's exception-suppressing
score postprocessor. No source rewriting or recovery of partial plots occurs.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def preflight():
    import matplotlib
    import numpy
    import pandas
    import scipy
    import pdf2image  # noqa: F401
    import PyPDF2  # noqa: F401
    assert sys.version_info[:2] == (3, 9)
    assert matplotlib.__version__ == "3.8.4"
    assert numpy.__version__ == "1.26.1"
    assert pandas.__version__ == "1.4.2"
    assert scipy.__version__ == "1.12.0"
    assert shutil.which("pdftoppm") and shutil.which("pdfinfo")


def render(source):
    from pdf2image import convert_from_path
    from pdf2image.exceptions import PDFPageCountError, PDFSyntaxError
    from PyPDF2 import PdfReader
    from PyPDF2.errors import PdfReadError

    pdf, preview = Path("/workspace/chart.pdf"), Path("/workspace/chart.png")
    if pdf.exists() or preview.exists():
        raise RuntimeError("Verification requires a fresh output directory")
    try:
        process = subprocess.run([sys.executable, str(source)], cwd="/workspace",
                                 env={**os.environ, "MPLBACKEND": "Agg"},
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
    except subprocess.TimeoutExpired:
        return 20, "candidate_timeout"
    if process.returncode < 0 or process.returncode in {137, 143}:
        raise RuntimeError("Candidate process interrupted by infrastructure")
    if process.returncode or not pdf.is_file() or pdf.is_symlink():
        return 20, "candidate_execution_failed"
    if not 0 < pdf.stat().st_size <= 32 * 1024 * 1024:
        return 20, "invalid_pdf_size"
    try:
        reader = PdfReader(str(pdf), strict=True)
        if len(reader.pages) != 1:
            return 20, "expected_single_page"
        page = reader.pages[0]
        width, height = float(page.mediabox.width), float(page.mediabox.height)
        if not 0 < width <= 3600 or not 0 < height <= 3600 or width * height > 8_000_000:
            return 20, "invalid_page_size"
        images = convert_from_path(pdf, dpi=120, first_page=1, last_page=1, timeout=45)
    except (PdfReadError, PDFPageCountError, PDFSyntaxError, ValueError):
        return 20, "invalid_pdf"
    if len(images) != 1:
        raise RuntimeError("PDF renderer produced unexpected pages")
    images[0].convert("RGB").save(preview, format="PNG")
    return 0, "ok"


def main():
    try:
        preflight()
        if sys.argv[1:] == ["--check"]:
            print("ChartMimic Python 3.9 / Matplotlib 3.8.4 ready")
            return 0
        if sys.argv[1:] != ["/workspace/answer.py"]:
            raise ValueError("Expected the fixed verification input")
        code, status = render(Path(sys.argv[1]))
    except Exception as exc:
        code, status = 30, "infrastructure_error:" + type(exc).__name__
    Path("/workspace/chart.result.json").write_text(json.dumps({"status": status, "exit_code": code}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
