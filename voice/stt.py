"""Speech-to-text via pluggable STTEngine (default: faster-whisper)."""

from __future__ import annotations

import gc
from dataclasses import dataclass

import numpy as np

from voice.config import get_sample_rate, get_stt_beam_size, get_stt_model_size
from voice.hotwords import apply_hotword_replacements
from voice.stt_engine import EngineLoadInfo, create_stt_engine, preload_engine, warmup_engine
from voice.stt_filters import apply_no_speech_filter

# Backward-compatible exports
ModelLoadInfo = EngineLoadInfo


@dataclass
class TranscriptionResult:
    text: str
    language_probability: float | None = None
    avg_no_speech_prob: float | None = None
    avg_logprob: float | None = None
    stt_model: str = ""
    stt_beam_size: int = 5
    audio_duration_sec: float | None = None


def get_whisper_model(
    model_size: str | None = None,
    compute_type: str | None = None,
    device: str = "cpu",
):
    """Legacy: return (engine, load_info) tuple."""
    engine = create_stt_engine(model_size=model_size, device=device, compute_type=compute_type)
    load_info = engine.load()
    return engine, load_info


def unload_stt_model() -> None:
    from voice.stt_engine import clear_engine_cache

    clear_engine_cache()
    gc.collect()


def preload_stt_model(model_size: str | None = None) -> ModelLoadInfo | None:
    engine = create_stt_engine(model_size=model_size)
    return preload_engine(engine)


def warmup_stt_model(model_size: str | None = None) -> None:
    engine = create_stt_engine(model_size=model_size)
    warmup_engine(engine, sample_rate=get_sample_rate())


def transcribe(audio: np.ndarray) -> str:
    result, _load = transcribe_with_metadata(audio)
    return result.text


def transcribe_with_metadata(
    audio: np.ndarray,
    *,
    word_timestamps: bool = False,
    beam_size: int | None = None,
) -> tuple[TranscriptionResult, ModelLoadInfo | None]:
    if audio is None or len(audio) == 0:
        return TranscriptionResult(text=""), None

    rate = get_sample_rate()
    duration_sec = len(audio) / float(rate)
    engine = create_stt_engine()
    load_info = engine.load()
    decoded = engine.transcribe(
        audio,
        word_timestamps=word_timestamps,
        beam_size=beam_size,
    )

    text = apply_hotword_replacements(decoded.text)
    text = apply_no_speech_filter(text, decoded.avg_no_speech_prob)

    result = TranscriptionResult(
        text=text,
        language_probability=decoded.language_probability,
        avg_no_speech_prob=decoded.avg_no_speech_prob,
        avg_logprob=decoded.avg_logprob,
        stt_model=get_stt_model_size(),
        stt_beam_size=beam_size if beam_size is not None else get_stt_beam_size(),
        audio_duration_sec=round(duration_sec, 3),
    )
    return result, load_info