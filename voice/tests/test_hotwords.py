import unittest

from voice.hotwords import apply_hotword_replacements


class TestHotwords(unittest.TestCase):
    def test_replacement(self) -> None:
        self.assertEqual(apply_hotword_replacements("run pie test"), "run pytest")


if __name__ == "__main__":
    unittest.main()