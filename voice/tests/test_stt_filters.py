import unittest

from voice.stt_filters import apply_no_speech_filter, is_likely_hallucination


class TestSttFilters(unittest.TestCase):
    def test_hallucination_phrase(self) -> None:
        self.assertTrue(is_likely_hallucination("thank you for watching", 0.5))

    def test_no_speech_empty(self) -> None:
        self.assertEqual(apply_no_speech_filter("hi", 0.9), "")


if __name__ == "__main__":
    unittest.main()