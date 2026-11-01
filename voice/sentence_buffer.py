"""Buffer streaming LLM tokens and emit speakable clauses."""

from voice.tts_sanitize import sanitize_for_tts

_CLAUSE_END = frozenset(".?!,")


class SentenceBuffer:
    """Accumulate token text; yield sanitized clauses at punctuation boundaries."""

    def __init__(self, min_clause_chars: int = 12):
        self._buf = ""
        self._min_clause_chars = min_clause_chars

    def push(self, token: str) -> list[str]:
        if not token:
            return []
        self._buf += token
        return self._drain(force=False)

    def flush(self) -> list[str]:
        return self._drain(force=True)

    def _drain(self, force: bool) -> list[str]:
        emitted: list[str] = []
        while True:
            chunk, rest = self._split_one(force)
            if chunk is None:
                break
            clean = sanitize_for_tts(chunk)
            if clean:
                emitted.append(clean)
            self._buf = rest
            force = False
        return emitted

    def _split_one(self, force: bool) -> tuple[str | None, str]:
        if not self._buf:
            return None, ""

        for idx, char in enumerate(self._buf):
            if char in _CLAUSE_END:
                end = idx + 1
                piece = self._buf[:end]
                if len(piece.strip()) >= self._min_clause_chars or char in ".?!":
                    return piece, self._buf[end:]

        if force and self._buf.strip():
            return self._buf, ""

        return None, self._buf
