"""Trim and normalize speech audio before STT."""

from __future__ import annotations

import numpy as np

from voice.config import (
    get_audio_highpass_enabled,
    get_audio_noise_suppress_enabled,
    get_audio_target_rms,
    get_sample_rate,
    get_speech_pad_ms,
)
from voice.vad import chunk_to_tensor, load_vad_model


def highpass_filter(audio: np.ndarray, sample_rate: int, cutoff_hz: float = 80.0) -> np.ndarray:
    """Simple first-order high-pass to reduce rumble."""
    if audio is None or len(audio) < 4:
        return audio
    rc = 1.0 / (2 * np.pi * cutoff_hz)
    dt = 1.0 / sample_rate
    alpha = rc / (rc + dt)
    y = np.zeros_like(audio, dtype=np.float32)
    y[0] = audio[0]
    for i in range(1, len(audio)):
        y[i] = alpha * (y[i - 1] + audio[i] - audio[i - 1])
    return y


def simple_noise_gate(audio: np.ndarray, floor: float = 0.008) -> np.ndarray:
    """Attenuate samples below a low energy floor (light suppression)."""
    if audio is None or len(audio) == 0:
        return audio
    out = audio.copy()
    mask = np.abs(out) < floor
    out[mask] *= 0.25
    return out


def preprocess_capture(audio: np.ndarray, sample_rate: int | None = None) -> np.ndarray:
    """Optional filters applied right after mic capture."""
    if audio is None or len(audio) == 0:
        return audio
    rate = sample_rate or get_sample_rate()
    out = audio.astype(np.float32)
    if get_audio_highpass_enabled():
        out = highpass_filter(out, rate)
    if get_audio_noise_suppress_enabled():
        out = simple_noise_gate(out)
    return out


def trim_to_speech(audio: np.ndarray, sample_rate: int | None = None) -> np.ndarray:
    """Keep speech region using Silero get_speech_timestamps with padding."""
    if audio is None or len(audio) == 0:
        return audio
    rate = sample_rate or get_sample_rate()
    pad_samples = int(get_speech_pad_ms() * rate / 1000)

    try:
        from silero_vad import get_speech_timestamps
    except ImportError:
        return audio

    model = load_vad_model()
    wav = chunk_to_tensor(audio)

    timestamps = get_speech_timestamps(
        wav,
        model,
        sampling_rate=rate,
        return_seconds=False,
    )
    if not timestamps:
        return audio

    start = max(0, timestamps[0]["start"] - pad_samples)
    end = min(len(audio), timestamps[-1]["end"] + pad_samples)
    if end <= start:
        return audio
    return audio[start:end].copy()


def normalize_rms(audio: np.ndarray, target_rms: float | None = None) -> np.ndarray:
    """Scale audio toward target RMS without clipping above 1.0."""
    if audio is None or len(audio) == 0:
        return audio
    target = target_rms if target_rms is not None else get_audio_target_rms()
    rms = float(np.sqrt(np.mean(np.square(audio))))
    if rms < 1e-6:
        return audio
    gain = target / rms
    out = audio * gain
    peak = float(np.max(np.abs(out)))
    if peak > 0.99:
        out = out * (0.99 / peak)
    return out.astype(np.float32)


def prepare_for_stt(audio: np.ndarray) -> tuple[np.ndarray, int, float]:
    """
    Trim silence padding then normalize.
    Returns (processed_audio, trimmed_sample_count, duration_sec).
    """
    rate = get_sample_rate()
    trimmed = trim_to_speech(audio, rate)
    normalized = normalize_rms(trimmed)
    duration_sec = len(normalized) / float(rate)
    return normalized, len(normalized), duration_sec