"""Hooks for a future full voice agent (LLM + TTS) on the voice event bus."""

from __future__ import annotations

from voice.brain import stream_reply
from voice.events import VoiceEventBus
from voice.tts import speak


class VoiceAgentBridge:
    """
    Stub integration layer for Phase 2+.
    Wire confirmed user text to Ollama and optionally speak replies.
    """

    def __init__(self, bus: VoiceEventBus, *, enable_tts: bool = False) -> None:
        self._bus = bus
        self._enable_tts = enable_tts
        self._history: list[dict] = []
        bus.subscribe_confirmed(self._on_confirmed)

    def _on_confirmed(self, text: str) -> None:
        if not text.strip():
            return
        reply_parts: list[str] = []
        for sentence in stream_reply(text, self._history):
            reply_parts.append(sentence)
            self._bus.emit_partial(sentence)
        reply = " ".join(reply_parts).strip()
        if reply:
            self._history.append({"role": "user", "content": text})
            self._history.append({"role": "assistant", "content": reply})
            self._bus.emit_final(reply)
            if self._enable_tts:
                speak(reply)