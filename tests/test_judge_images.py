import io
import unittest
from unittest.mock import patch

from PIL import Image

from mmrl.judge_images import bounded_image_bytes, fit_size, patch_count
from mmrl.reward import InfrastructureError, image_data_uri


class JudgeImageTests(unittest.TestCase):
    def test_real_rejected_dimensions_fit_without_cropping(self):
        self.assertEqual(patch_count(11521, 8401), 94943)
        for width, height in [(11521, 8401), (9600, 8401), (8401, 11521), (32 * 30000 + 1, 1)]:
            with self.subTest(size=(width, height)):
                w, h = fit_size(width, height)
                self.assertLessEqual(patch_count(w, h), 30000)
                self.assertLessEqual(w, width)
                self.assertLessEqual(h, height)
                self.assertLessEqual(abs(w * height - h * width), max(width, height))

    def test_small_inputs_are_byte_identical(self):
        for fmt in ("PNG", "JPEG", "WEBP"):
            stream = io.BytesIO()
            Image.new("RGB", (64, 32), "red").save(stream, format=fmt)
            data = stream.getvalue()
            self.assertEqual(bounded_image_bytes(data), data)

    def test_resize_preserves_whole_image_and_alpha(self):
        im = Image.new("RGBA", (128, 64), (255, 0, 0, 128))
        im.paste((0, 0, 255, 128), (96, 0, 128, 64))
        stream = io.BytesIO()
        im.save(stream, format="PNG")
        with patch("mmrl.judge_images.MAX_PATCHES", 2):
            result = bounded_image_bytes(stream.getvalue())
        with Image.open(io.BytesIO(result)) as out:
            self.assertLessEqual(patch_count(*out.size), 2)
            self.assertEqual(out.mode, "RGBA")
            self.assertEqual(out.getpixel((0, 0))[0], 255)
            self.assertEqual(out.getpixel((out.width - 1, 0))[2], 255)
            self.assertEqual(out.getpixel((0, 0))[3], 128)

    def test_malformed_evidence_is_infrastructure_failure(self):
        with self.assertRaises(InfrastructureError):
            image_data_uri(b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
