import unittest

from voice.transcript_confirm import prompt_transcript_edit


class TranscriptConfirmTests(unittest.TestCase):
    def test_accept_empty_line_keeps_transcript(self):
        result = prompt_transcript_edit("hello world", input_fn=lambda _: "")
        self.assertEqual(result, "hello world")

    def test_edit_replaces_transcript(self):
        result = prompt_transcript_edit("helo wrld", input_fn=lambda _: "hello world")
        self.assertEqual(result, "hello world")

    def test_cancel_returns_none(self):
        result = prompt_transcript_edit("test", input_fn=lambda _: ".cancel")
        self.assertIsNone(result)
