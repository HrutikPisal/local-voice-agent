"""Microphone capture with Silero VAD endpointing."""

from __future__ import annotations

import collections
import threading
import time
from collections.abc import Callable

import numpy as np

from voice.audio_prep import preprocess_capture
from voice.config import (
    get_input_device,
    get_max_record_seconds,
    get_record_preroll_ms,
    get_sample_rate,
    get_vad_chunk_samples,
)
from voice.endpointing import AdaptiveEndpointTracker
from voice.streaming_stt import StreamingPartialSTT
from voice.vad import create_vad_iterator, process_vad_chunk, speech_ended


class MicLevelMeter:
    def __init__(self, enabled: bool = True) -> None:
        self._enabled = enabled
        self._last_print = 0.0

    def update(self, chunk_int16: np.ndarray) -> None:
        if not self._enabled:
            return
        now = time.monotonic()
        if now - self._last_print < 0.35:
            return
        self._last_print = now
        peak = float(np.max(np.abs(chunk_int16))) / 32768.0
        bars = int(min(peak, 1.0) * 20)
        print(f"\r[mic level] {'|' * bars}{' ' * (20 - bars)} {peak:0.2f}   ", end="", flush=True)


def capture_utterance(
    manual_stop: threading.Event | None = None,
    cancel: threading.Event | None = None,
    *,
    device: int | None = None,
    on_partial: Callable[[str], None] | None = None,
    endpoint_tracker: AdaptiveEndpointTracker | None = None,
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
    preroll_max = max(1, int(get_record_preroll_ms() * sample_rate / 1000 / chunk_samples))
    preroll: collections.deque[np.ndarray] = collections.deque(maxlen=preroll_max)
    blocks: list[np.ndarray] = []
    speech_seen = False
    vad_iterator = create_vad_iterator()
    deadline = time.monotonic() + get_max_record_seconds()
    dev = device if device is not None else get_input_device()
    tracker = endpoint_tracker or AdaptiveEndpointTracker()
    meter = MicLevelMeter()

    partial_stt: StreamingPartialSTT | None = None
    if on_partial is not None:

        def _partial_cb(text: str) -> None:
            tracker.update_partial(text)
            on_partial(text)

        partial_stt = StreamingPartialSTT(_partial_cb)
        partial_stt.start()

    vad_end_at: float | None = None

    def aborted() -> bool:
        if cancel and cancel.is_set():
            return True
        if manual_stop and manual_stop.is_set():
            return True
        return time.monotonic() >= deadline

    try:
        stream_kw = dict(
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
            blocksize=chunk_samples,
        )
        if dev is not None:
            stream_kw["device"] = dev
        with sd.InputStream(**stream_kw) as stream:
            while not aborted():
                chunk, _overflowed = stream.read(chunk_samples)
                flat = np.asarray(chunk, dtype=np.int16).reshape(-1)
                if len(flat) < chunk_samples:
                    pad = np.zeros(chunk_samples - len(flat), dtype=np.int16)
                    flat = np.concatenate([flat, pad])

                preroll.append(flat.copy())
                meter.update(flat)
                if partial_stt is not None:
                    partial_stt.push_chunk(flat)

                if speech_seen:
                    blocks.append(flat.copy())

                event = process_vad_chunk(vad_iterator, flat)
                if event and "start" in event:
                    speech_seen = True
                    blocks = list(preroll)
                    vad_end_at = None

                if speech_seen and speech_ended(event):
                    vad_end_at = time.monotonic()

                if vad_end_at is not None:
                    extra_s = tracker.extra_silence_ms_after_speech_end() / 1000.0
                    if time.monotonic() - vad_end_at >= extra_s:
                        break

                if manual_stop and manual_stop.is_set() and speech_seen:
                    break
    except Exception as exc:
        raise RuntimeError(f"Microphone capture failed: {exc}") from exc
    finally:
        print("\r" + " " * 40 + "\r", end="", flush=True)
        if partial_stt is not None:
            partial_stt.stop()

    if cancel and cancel.is_set():
        return None
    if not blocks:
        return None
    manual = manual_stop and manual_stop.is_set()
    if not speech_seen and not manual:
        return None

    pcm = np.concatenate(blocks)
    pcm = pcm.astype(np.float32) / 32768.0
    return preprocess_capture(pcm, sample_rate)