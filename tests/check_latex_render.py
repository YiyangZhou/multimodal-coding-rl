"""Test the LaTeX body contract using the real disposable-sandbox renderer."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from PIL import Image, ImageStat

RENDERER = Path(os.environ.get("MMRL_RENDER_SCRIPT", Path(__file__).resolve().parents[1] / "mmrl/render.py"))


class LatexRenderTest(unittest.TestCase):
    def run_case(self, body, succeeds=True):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "answer.tex", root / "preview.png"
            source.write_text(body)
            result = subprocess.run([sys.executable, str(RENDERER), "image2struct_latex",
                                     str(source), str(output)], capture_output=True,
                                    text=True, timeout=180)
            self.assertEqual(source.read_text(), body, "Authored answer must remain unchanged")
            if not succeeds:
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(output.exists())
                return
            self.assertEqual(result.returncode, 0, result.stderr[-2000:])
            with Image.open(output) as image:
                self.assertGreater(max(ImageStat.Stat(image.convert("RGB")).var), 0)

    def test_equation_body(self):
        self.run_case(r"\begin{equation*}x^2+y^2=z^2\end{equation*}")

    def test_box_body(self):
        self.run_case(r"\begin{tcolorbox}[colback=white,colframe=black,arc=0mm]Algorithm $x=1$.\end{tcolorbox}")

    def test_math_command_outside_math_still_fails(self):
        self.run_case(r"\mathbb{P}_{\alpha,n}", succeeds=False)

    def test_algorithm_uppercase(self):
        self.run_case(r"\begin{algorithm}\caption{Synthetic test}\begin{algorithmic}[1]\STATE $x\gets 1$\IF{$x>0$}\STATE $x\gets 2$\ENDIF\end{algorithmic}\end{algorithm}")

    def test_algorithm_mixed_case(self):
        self.run_case(r"\begin{algorithm}\caption{Synthetic test}\begin{algorithmic}[1]\State $x\gets 1$\If{$x>0$}\State $x\gets 2$\EndIf\end{algorithmic}\end{algorithm}")

    def test_table_body(self):
        self.run_case(r"\begin{tabular}{cc}\toprule A&B\\\midrule 1&2\\\bottomrule\end{tabular}")

    def test_tikz_body(self):
        self.run_case(r"\begin{tikzpicture}\draw (0,0) circle (1cm);\end{tikzpicture}")

    def test_full_document(self):
        self.run_case(r"\documentclass{article}\begin{document}Synthetic document.\end{document}")

    def test_comment_is_not_a_document(self):
        self.run_case("% \\documentclass{article}\n" + r"\begin{equation*}x=1\end{equation*}")

    def test_invalid_body_still_fails(self):
        self.run_case(r"\undefinedSyntheticCommand", succeeds=False)

    def test_incomplete_explicit_document_still_fails(self):
        self.run_case(r"\documentclass{article}\begin{document}Missing end", succeeds=False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
