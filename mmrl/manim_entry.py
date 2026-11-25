"""Manim CLI with the pinned public asset; candidate/reference sources stay intact."""

import os
import runpy
import sys
import traceback

from manim_assets import install_asset_cache


def install_numpy_compat():
    """Keep Manim 0.19's trapezoid integration working with NumPy 2.4+."""
    import numpy as np

    # NumPy removed the deprecated name; trapezoid is its replacement.
    # Apply equally to candidate and reference renderers, leaving source intact.
    if not hasattr(np, "trapz"):
        np.trapz = np.trapezoid


if __name__ == "__main__":
    install_asset_cache()
    install_numpy_compat()
    try:
        runpy.run_module("manim", run_name="__main__")
    except SystemExit as exc:
        if exc.code in (None, 0):
            raise
        # The CLI may already print its traceback and exit via Click.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exc.code if isinstance(exc.code, int) and 1 <= exc.code <= 255 else 1)
    except Exception:
        # An uncaught scene error can leave Manim's non-daemon writer waiting
        # for frames forever. Preserve the error, then exit this disposable
        # renderer process without waiting for that broken writer.
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
