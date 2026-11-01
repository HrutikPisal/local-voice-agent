import unittest

from voice.tts_sanitize import sanitize_for_tts


class TtsSanitizeTests(unittest.TestCase):
    def test_strips_markdown(self):
        raw = "**Hello** see [link](https://x.com) and `code`"
        clean = sanitize_for_tts(raw)
        self.assertNotIn("**", clean)
        self.assertNotIn("`", clean)
        self.assertIn("Hello", clean)
        self.assertIn("link", clean)

    def test_empty(self):
        self.assertEqual(sanitize_for_tts(""), "")
