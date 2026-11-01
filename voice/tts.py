"""Text-to-speech via Piper."""

from __future__ import annotations

import io
import threading
import wave
from collections.abc import Iterable

import numpy as np

from voice.config import piper_model_paths
from voice.setup import ensure_models
from voice.tts_sanitize import sanitize_for_tts

_voice = None


def _load_voice():
    global _voice
    if _voice is not None:
        return _voice
    ensure_models()
    try:
        from piper import PiperVoice
    except ImportError as exc:
        raise ImportError(
            "piper-tts is required. Install with: pip install -r voice/requirements.txt"
        ) from exc
    model_path, config_path = piper_model_paths()
    _voice = PiperVoice.load(str(model_path), config_path=str(config_path))
    return _voice


def unload_voice() -> None:
    global _voice
    _voice = None


def _play_pcm(pcm_bytes: bytes, sample_rate: int, stop_event: threading.Event | None) -> None:
    import sounddevice as sd

    if stop_event and stop_event.is_set():
        return
    samples = np.frombuffer(pcm_bytes, dtype=np.int16)
    if len(samples) == 0:
        return
    sd.play(samples, sample_rate)
    duration = len(samples) / float(sample_rate)
    waited = 0.0
    while waited < duration + 0.25:
        if stop_event and stop_event.is_set():
            sd.stop()
            break
        sd.sleep(50)
        waited += 0.05


def _synthesize_chunks(text: str) -> Iterable[tuple[bytes, int]]:
    voice = _load_voice()
    clean = sanitize_for_tts(text)
    if not clean:
        return

    try:
        from piper import SynthesisConfig

        for chunk in voice.synthesize(clean, SynthesisConfig()):
            yield chunk.audio_int16_bytes, chunk.sample_rate
        return
    except (ImportError, TypeError):
        pass

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        voice.synthesize(clean, wav_file)
    buffer.seek(0)
    with wave.open(buffer, "rb") as wav_read:
        rate = wav_read.getframerate()
        frames = wav_read.readframes(wav_read.getnframes())
        yield frames, rate


def speak(text: str, stop_event: threading.Event | None = None) -> None:
    """Speak full text; honor stop_event during playback."""
    for pcm, rate in _synthesize_chunks(text):
        _play_pcm(pcm, rate, stop_event)
        if stop_event and stop_event.is_set():
            break


def speak_clauses(clauses: Iterable[str], stop_event: threading.Event | None = None) -> None:
    for clause in clauses:
        if stop_event and stop_event.is_set():
            break
        speak(clause, stop_event=stop_event)
