"""Minimal streaming Ollama chat for the voice demo (no tools)."""

from __future__ import annotations

import threading
from collections.abc import Iterator

import ollama

from voice.config import get_default_ollama_model, get_ollama_num_predict
from voice.prompt import get_voice_system_prompt
from voice.sentence_buffer import SentenceBuffer
from voice.telemetry import TurnTelemetry


def stream_reply(
    user_text: str,
    history: list[dict],
    stop_event: threading.Event | None = None,
    telemetry: TurnTelemetry | None = None,
) -> Iterator[str]:
    """Stream assistant tokens from Ollama."""
    messages = [{"role": "system", "content": get_voice_system_prompt()}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_text})

    stream = ollama.chat(
        model=get_default_ollama_model(),
        messages=messages,
        stream=True,
        options={"num_predict": get_ollama_num_predict()},
    )
    try:
        for chunk in stream:
            if stop_event and stop_event.is_set():
                break
            content = chunk.get("message", {}).get("content") or ""
            if content:
                if telemetry:
                    telemetry.mark_llm_first_token()
                yield content
    finally:
        close = getattr(stream, "close", None)
        if callable(close):
            close()


def stream_clauses_for_tts(
    user_text: str,
    history: list[dict],
    stop_event: threading.Event | None = None,
    telemetry: TurnTelemetry | None = None,
) -> tuple[list[str], str]:
    """
    Stream LLM output and collect speakable clauses plus full reply text.
    """
    buffer = SentenceBuffer()
    full_parts: list[str] = []
    all_clauses: list[str] = []

    for token in stream_reply(user_text, history, stop_event, telemetry):
        full_parts.append(token)
        all_clauses.extend(buffer.push(token))

    all_clauses.extend(buffer.flush())
    full_text = sanitize_join("".join(full_parts))
    return all_clauses, full_text


def sanitize_join(text: str) -> str:
    return text.strip()


def chat_turn(
    user_text: str,
    history: list[dict],
    *,
    stream_tts: bool = True,
    stop_event: threading.Event | None = None,
    telemetry: TurnTelemetry | None = None,
    speak_fn=None,
) -> str:
    """
    Run one LLM turn. If stream_tts and speak_fn provided, speak clause-by-clause.
    speak_fn signature: (clause: str) -> None
    """
    if stream_tts and speak_fn:
        clauses, full = stream_clauses_for_tts(user_text, history, stop_event, telemetry)
        for clause in clauses:
            if stop_event and stop_event.is_set():
                break
            speak_fn(clause)
        return full

    parts: list[str] = []
    for token in stream_reply(user_text, history, stop_event, telemetry):
        parts.append(token)
    return "".join(parts).strip()
