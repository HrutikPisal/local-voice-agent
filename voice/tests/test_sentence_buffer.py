import unittest

from voice.sentence_buffer import SentenceBuffer


class SentenceBufferTests(unittest.TestCase):
    def test_emits_on_sentence_end(self):
        buf = SentenceBuffer(min_clause_chars=3)
        parts = []
        for ch in "Hello there.":
            parts.extend(buf.push(ch))
        parts.extend(buf.flush())
        joined = " ".join(parts)
        self.assertIn("Hello", joined)

    def test_flush_remainder(self):
        buf = SentenceBuffer()
        buf.push("Short reply")
        tail = buf.flush()
        self.assertTrue(tail)
        self.assertIn("Short reply", tail[0])
