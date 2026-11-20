import unittest

from voice.transcript_confirm import TranscriptConfidence, should_auto_accept


class TestAutoAccept(unittest.TestCase):
    def test_disabled_by_default(self) -> None:
        conf = TranscriptConfidence(avg_logprob=0.0, avg_no_speech_prob=0.1)
        self.assertFalse(should_auto_accept(conf))


if __name__ == "__main__":
    unittest.main()