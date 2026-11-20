import unittest

import numpy as np

from voice.audio_prep import highpass_filter, preprocess_capture


class TestAudioPrepFilters(unittest.TestCase):
    def test_highpass_preserves_length(self) -> None:
        audio = np.random.randn(1600).astype(np.float32) * 0.01
        out = highpass_filter(audio, 16000)
        self.assertEqual(len(out), len(audio))

    def test_preprocess_capture(self) -> None:
        audio = np.zeros(800, dtype=np.float32)
        out = preprocess_capture(audio)
        self.assertEqual(len(out), 800)


if __name__ == "__main__":
    unittest.main()