import os
import unittest
from unittest import mock

from voice.transcript_confirm import (
    prompt_transcript_edit,
    set_force_plain_confirm,
    should_use_plain_confirm,
)


class TranscriptConfirmTests(unittest.TestCase):
    def tearDown(self) -> None:
        set_force_plain_confirm(None)

    def test_accept_empty_line_keeps_transcript(self):
        result = prompt_transcript_edit("hello world", input_fn=lambda _: "")
        self.assertEqual(result, "hello world")

    def test_edit_replaces_transcript(self):
        result = prompt_transcript_edit("helo wrld", input_fn=lambda _: "hello world")
        self.assertEqual(result, "hello world")

    def test_cancel_returns_none(self):
        result = prompt_transcript_edit("test", input_fn=lambda _: ".cancel")
        self.assertIsNone(result)

    def test_plain_confirm_path(self):
        result = prompt_transcript_edit(
            "alpha",
            input_fn=lambda _: "beta",
            plain_confirm=True,
        )
        self.assertEqual(result, "beta")

    def test_should_use_plain_confirm_vscode(self):
        with mock.patch.dict(os.environ, {"TERM_PROGRAM": "vscode"}, clear=False):
            set_force_plain_confirm(None)
            self.assertTrue(should_use_plain_confirm())

    def test_should_use_plain_confirm_env(self):
        with mock.patch.dict(os.environ, {"VOICE_PLAIN_CONFIRM": "1"}, clear=False):
            set_force_plain_confirm(None)
            self.assertTrue(should_use_plain_confirm())

    def test_force_rich_overrides_vscode(self):
        with mock.patch.dict(os.environ, {"TERM_PROGRAM": "vscode"}, clear=False):
            set_force_plain_confirm(False)
            self.assertFalse(should_use_plain_confirm())


if __name__ == "__main__":
    unittest.main()