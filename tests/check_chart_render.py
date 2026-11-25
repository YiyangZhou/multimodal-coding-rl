"""Exercise the actual renderer subprocess in a disposable CPU sandbox."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from PIL import Image, ImageChops, ImageStat

RENDERER = Path(os.environ.get("MMRL_RENDER_SCRIPT", Path(__file__).resolve().parents[1] / "mmrl/render.py"))
SETUP = """import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(4, 3))
ax.plot([0, 1, 2], [0, 2, 1], color='red', linewidth=5)
ax.set_title('Synthetic close regression')
"""
SAVE = "fig.savefig('authored.png', dpi=120, bbox_inches='tight')\n"


class ChartRenderTest(unittest.TestCase):
    def run_case(self, code, *, succeeds=True, expected="authored.png"):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            source, output = directory / "answer.py", directory / "result.png"
            source.write_text(code)
            result = subprocess.run(
                [sys.executable, str(RENDERER), "chart2code", str(source), str(output)],
                capture_output=True, text=True, timeout=150,
            )
            if not succeeds:
                self.assertNotEqual(result.returncode, 0)
                return
            self.assertEqual(result.returncode, 0, result.stderr[-2000:])
            with Image.open(output) as image:
                actual = image.convert("RGB")
            self.assertGreater(max(ImageStat.Stat(actual).var), 100)
            if expected:
                with Image.open(directory / expected) as image:
                    reference = image.convert("RGB")
                self.assertEqual(actual.size, reference.size)
                self.assertIsNone(ImageChops.difference(actual, reference).getbbox())

    def test_saved_then_closed(self):
        self.run_case(SETUP + SAVE + "plt.close(fig)\n")

    def test_saved_then_closed_all(self):
        self.run_case(SETUP + SAVE + "plt.close('all')\n")

    def test_saved_then_cleared(self):
        self.run_case(SETUP + SAVE + "fig.clear()\n")

    def test_last_explicit_save_wins(self):
        self.run_case(SETUP + "fig.savefig('earlier.png')\nax.set_title('Final')\n" + SAVE + "plt.close('all')\n")

    def test_saved_figure_not_unsaved_new_figure(self):
        self.run_case(SETUP + SAVE + "plt.figure()\nplt.plot([1, 2], [9, 0])\n")

    def test_show_only(self):
        self.run_case(SETUP + "plt.show()\n", expected=None)

    def test_closed_without_save(self):
        self.run_case(SETUP + "plt.close(fig)\n", expected=None)

    def test_active_without_save(self):
        self.run_case(SETUP, expected=None)

    def test_direct_figure_save(self):
        self.run_case("from matplotlib.figure import Figure\nfig=Figure(figsize=(4,3))\n"
                      "ax=fig.subplots()\nax.plot([0,1,2],[0,2,1])\n" + SAVE + "fig.clear()\n")

    def test_no_figure_is_execution_failure(self):
        self.run_case("value = 1 + 2\n", succeeds=False)

    def test_error_after_save_still_fails(self):
        self.run_case(SETUP + SAVE + "raise RuntimeError('synthetic failure')\n", succeeds=False)

    def test_missing_save_directory_still_fails(self):
        self.run_case(SETUP + "fig.savefig('absent/plot.png')\n", succeeds=False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
