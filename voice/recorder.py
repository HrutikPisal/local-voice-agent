"""Microphone capture with Silero VAD endpointing."""

from __future__ import annotations

import threading
import time

import numpy as np

from voice.config import get_max_record_seconds, get_sample_rate, get_vad_chunk_samples
from voice.vad import create_vad_iterator, process_vad_chunk, speech_ended


def capture_utterance(
    manual_stop: threading.Event | None = None,
    cancel: threading.Event | None = None,
) -> np.ndarray | None:
    """
    Record mono audio at 16 kHz until VAD endpoint, manual stop, cancel, or max duration.
    Returns float32 samples in [-1, 1], or None if cancelled / no input device.
    """
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise ImportError(
            "sounddevice is required. Install with: pip install -r voice/requirements.txt"
        ) from exc

    sample_rate = get_sample_rate()
    chunk_samples = get_vad_chunk_samples()
    max_samples = int(get_max_record_seconds() * sample_rate)
    blocks: list[np.ndarray] = []
    speech_seen = False
    vad_iterator = create_vad_iterator()
    deadline = time.monotonic() + get_max_record_seconds()

    def aborted() -> bool:
        if cancel and cancel.is_set():
            return True
        if manual_stop and manual_stop.is_set():
            return True
        return time.monotonic() >= deadline

    try:
        with sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
            blocksize=chunk_samples,
        ) as stream:
            while not aborted():
                chunk, _overflowed = stream.read(chunk_samples)
                flat = np.asarray(chunk, dtype=np.int16).reshape(-1)
                if len(flat) < chunk_samples:
                    pad = np.zeros(chunk_samples - len(flat), dtype=np.int16)
                    flat = np.concatenate([flat, pad])
                blocks.append(flat.copy())
                event = process_vad_chunk(vad_iterator, flat)
                if event and "start" in event:
                    speech_seen = True
                if speech_seen and speech_ended(event):
                    break
                if manual_stop and manual_stop.is_set() and speech_seen:
                    break
    except Exception as exc:
        raise RuntimeError(f"Microphone capture failed: {exc}") from exc

    if cancel and cancel.is_set():
        return None
    if not blocks:
        return None
    manual = manual_stop and manual_stop.is_set()
    if not speech_seen and not manual:
        return None

    pcm = np.concatenate(blocks)
    pcm = pcm.astype(np.float32) / 32768.0
    return pcm
