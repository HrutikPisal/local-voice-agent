import unittest

import numpy as np

from voice.audio_prep import normalize_rms, trim_to_speech


class AudioPrepTests(unittest.TestCase):
    def test_normalize_rms_scales_quiet_signal(self):
        quiet = np.full(1600, 0.001, dtype=np.float32)
        out = normalize_rms(quiet, target_rms=0.08)
        rms = float(np.sqrt(np.mean(np.square(out))))
        self.assertAlmostEqual(rms, 0.08, places=2)
        self.assertLessEqual(float(np.max(np.abs(out))), 0.99)

    def test_trim_empty_returns_empty(self):
        empty = np.array([], dtype=np.float32)
        self.assertEqual(len(trim_to_speech(empty)), 0)


if __name__ == "__main__":
    unittest.main()
