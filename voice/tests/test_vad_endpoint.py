import unittest

from voice.vad import speech_ended


class VadEndpointTests(unittest.TestCase):
    def test_speech_ended_on_end_key(self):
        self.assertTrue(speech_ended({"start": 0, "end": 100}))
        self.assertFalse(speech_ended({"start": 0}))
        self.assertFalse(speech_ended(None))
