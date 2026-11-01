"""Silero VAD loading and streaming endpoint detection."""

from __future__ import annotations

import numpy as np

from voice.config import (
    get_sample_rate,
    get_vad_chunk_samples,
    get_vad_min_silence_ms,
    get_vad_threshold,
)

_vad_model = None


def load_vad_model():
    """Lazy-load Silero VAD (ONNX when available)."""
    global _vad_model
    if _vad_model is not None:
        return _vad_model
    try:
        from silero_vad import load_silero_vad
    except ImportError as exc:
        raise ImportError(
            "silero-vad is required for voice recording. "
            "Install with: pip install -r voice/requirements.txt"
        ) from exc
    _vad_model = load_silero_vad(onnx=True)
    return _vad_model


def reset_vad_cache() -> None:
    global _vad_model
    _vad_model = None


def create_vad_iterator():
    from silero_vad import VADIterator

    model = load_vad_model()
    return VADIterator(
        model,
        threshold=get_vad_threshold(),
        sampling_rate=get_sample_rate(),
        min_silence_duration_ms=get_vad_min_silence_ms(),
    )


def chunk_to_tensor(chunk: np.ndarray):
    import torch

    if chunk.dtype == np.int16:
        floats = chunk.astype(np.float32) / 32768.0
    else:
        floats = chunk.astype(np.float32)
    return torch.from_numpy(floats)


def process_vad_chunk(vad_iterator, chunk: np.ndarray) -> dict | None:
    """Feed one fixed-size chunk; return VAD event dict when speech ends."""
    chunk_samples = get_vad_chunk_samples()
    if len(chunk) != chunk_samples:
        return None
    tensor = chunk_to_tensor(chunk)
    return vad_iterator(tensor, return_seconds=False)


def speech_ended(event: dict | None) -> bool:
    if not event:
        return False
    return "end" in event
