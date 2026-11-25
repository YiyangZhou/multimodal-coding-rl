"""Keep diff anchors faithful to the original fenced document."""
import unittest
from mmrl.webmmu_patch import apply_webmmu_patch

class FencedSourceTests(unittest.TestCase):
    source = '```html\n<p>old</p>\n```'

    def test_diff_can_anchor_literal_source_wrapper(self):
        diff = ' ```html\n-<p>old</p>\n+<p>new</p>\n ```'
        self.assertEqual(apply_webmmu_patch(self.source, diff), '```html\n<p>new</p>\n```')

    def test_body_only_diff_still_works(self):
        self.assertEqual(apply_webmmu_patch(self.source, '-<p>old</p>\n+<p>new</p>'), '<p>new</p>')

    def test_wrapper_does_not_allow_wrong_original_content(self):
        with self.assertRaises(ValueError):
            apply_webmmu_patch(self.source, ' ```html\n-<p>different</p>\n+<p>new</p>\n ```')

if __name__ == '__main__':
    unittest.main()
