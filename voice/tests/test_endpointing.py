import unittest

from voice.endpointing import AdaptiveEndpointTracker


class TestAdaptiveEndpoint(unittest.TestCase):
    def test_filler_extends_wait(self) -> None:
        t = AdaptiveEndpointTracker()
        t.update_partial("run git commit and")
        self.assertGreater(t.extra_silence_ms_after_speech_end(), 0)

    def test_sentence_shortens_min_silence(self) -> None:
        t = AdaptiveEndpointTracker()
        t.update_partial("run pytest now.")
        self.assertLessEqual(t.effective_min_silence_ms(), 400)


if __name__ == "__main__":
    unittest.main()