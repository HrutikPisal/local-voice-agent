import unittest

import numpy as np


class SttTests(unittest.TestCase):
    def test_empty_audio_skips_model(self):
        from voice.stt import transcribe

        self.assertEqual(transcribe(np.array([], dtype=np.float32)), "")


class MissingDepsTests(unittest.TestCase):
    def test_main_reports_missing_deps(self):
        from voice.main import _missing_deps_message

        msg = _missing_deps_message()
        if msg:
            self.assertIn("requirements-stt", msg)
