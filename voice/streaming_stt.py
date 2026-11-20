"""Background partial STT while the user is still speaking."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

import numpy as np

from voice.audio_prep import normalize_rms
from voice.config import (
  get_sample_rate,
  get_stt_partial_beam_size,
  get_stt_partial_enabled,
  get_stt_partial_interval_sec,
  get_stt_partial_min_audio_sec,
)
from voice.stt_engine import STTEngine, create_stt_engine


class StreamingPartialSTT:
  """Decodes rolling audio windows on a worker thread."""

  def __init__(
    self,
    on_partial: Callable[[str], None],
    engine: STTEngine | None = None,
  ) -> None:
    self._on_partial = on_partial
    self._engine = engine
    self._lock = threading.Lock()
    self._blocks: list[np.ndarray] = []
    self._stop = threading.Event()
    self._thread: threading.Thread | None = None
    self._last_emitted = ""

  def start(self) -> None:
    if not get_stt_partial_enabled():
      return
    self._stop.clear()
    self._thread = threading.Thread(target=self._worker, daemon=True)
    self._thread.start()

  def push_chunk(self, chunk_int16: np.ndarray) -> None:
    if not get_stt_partial_enabled():
      return
    floats = chunk_int16.astype(np.float32) / 32768.0
    with self._lock:
      self._blocks.append(floats.copy())

  def stop(self) -> None:
    self._stop.set()
    if self._thread and self._thread.is_alive():
      self._thread.join(timeout=2.0)

  def _concat_audio(self) -> np.ndarray | None:
    with self._lock:
      if not self._blocks:
        return None
      return np.concatenate(self._blocks)

  def _worker(self) -> None:
    rate = get_sample_rate()
    min_samples = int(get_stt_partial_min_audio_sec() * rate)
    interval = get_stt_partial_interval_sec()
    partial_engine = self._engine
    if partial_engine is None:
      partial_engine = create_stt_engine()

    while not self._stop.is_set():
      time.sleep(interval)
      audio = self._concat_audio()
      if audio is None or len(audio) < min_samples:
        continue
      try:
        prepared = normalize_rms(audio)
        result = partial_engine.transcribe(prepared, beam_size=get_stt_partial_beam_size())
        text = (result.text or "").strip()
      except Exception:
        continue
      if not text or text == self._last_emitted:
        continue
      self._last_emitted = text
      self._on_partial(text)