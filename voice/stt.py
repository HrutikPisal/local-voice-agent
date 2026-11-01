"""Speech-to-text via faster-whisper (cached model per process)."""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass

import numpy as np

from voice.config import (
    get_sample_rate,
    get_stt_beam_size,
    get_stt_compute_type,
    get_stt_condition_on_previous_text,
    get_stt_initial_prompt,
    get_stt_language,
    get_stt_model_size,
    get_stt_vad_filter,
)

_model_cache: dict[tuple[str, str, str], object] = {}


@dataclass
class TranscriptionResult:
    text: str
    language_probability: float | None = None
    avg_no_speech_prob: float | None = None
    stt_model: str = ""
    stt_beam_size: int = 5
    audio_duration_sec: float | None = None


@dataclass
class ModelLoadInfo:
    stt_model_load_ms: float
    stt_model: str
    compute_type: str
    device: str = "cpu"


def _cache_key(model_size: str, compute_type: str, device: str) -> tuple[str, str, str]:
    return (model_size, compute_type, device)


def get_whisper_model(
    model_size: str | None = None,
    compute_type: str | None = None,
    device: str = "cpu",
) -> tuple[object, ModelLoadInfo | None]:
    """Return cached WhisperModel. ModelLoadInfo is set only on first load in this process."""
    size = model_size or get_stt_model_size()
    ctype = compute_type or get_stt_compute_type()
    key = _cache_key(size, ctype, device)

    load_info: ModelLoadInfo | None = None
    if key not in _model_cache:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise ImportError(
                "faster-whisper is required. Install with: pip install -r voice/requirements-stt.txt"
            ) from exc
        started = time.perf_counter()
        _model_cache[key] = WhisperModel(size, device=device, compute_type=ctype)
        load_ms = (time.perf_counter() - started) * 1000
        load_info = ModelLoadInfo(
            stt_model_load_ms=load_ms,
            stt_model=size,
            compute_type=ctype,
            device=device,
        )
    return _model_cache[key], load_info


def unload_stt_model() -> None:
    """Free cached Whisper models (RAM emergency)."""
    global _model_cache
    _model_cache.clear()
    gc.collect()


def transcribe(audio: np.ndarray) -> str:
    """Transcribe float32 mono PCM; returns text only (backward compatible)."""
    result, _load = transcribe_with_metadata(audio)
    return result.text


def transcribe_with_metadata(
    audio: np.ndarray,
) -> tuple[TranscriptionResult, ModelLoadInfo | None]:
    """Transcribe and return text plus confidence metadata for audit logs."""
    if audio is None or len(audio) == 0:
        return TranscriptionResult(text=""), None

    rate = get_sample_rate()
    duration_sec = len(audio) / float(rate)
    model, load_info = get_whisper_model()

    initial_prompt = get_stt_initial_prompt()
    prompt_kw = {"initial_prompt": initial_prompt} if initial_prompt else {}

    segments, info = model.transcribe(
        audio,
        language=get_stt_language(),
        beam_size=get_stt_beam_size(),
        vad_filter=get_stt_vad_filter(),
        condition_on_previous_text=get_stt_condition_on_previous_text(),
        word_timestamps=False,
        **prompt_kw,
    )

    seg_list = list(segments)
    parts = [seg.text.strip() for seg in seg_list if seg.text.strip()]
    text = " ".join(parts).strip()

    no_speech_probs = [
        seg.no_speech_prob for seg in seg_list if getattr(seg, "no_speech_prob", None) is not None
    ]
    avg_no_speech = sum(no_speech_probs) / len(no_speech_probs) if no_speech_probs else None
    lang_prob = getattr(info, "language_probability", None)

    result = TranscriptionResult(
        text=text,
        language_probability=lang_prob,
        avg_no_speech_prob=avg_no_speech,
        stt_model=get_stt_model_size(),
        stt_beam_size=get_stt_beam_size(),
        audio_duration_sec=round(duration_sec, 3),
    )
    return result, load_info
