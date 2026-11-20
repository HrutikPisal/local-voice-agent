"""Adaptive end-of-turn timing on top of Silero VAD."""

from __future__ import annotations

import re

from voice.config import (
  get_vad_adaptive_enabled,
  get_vad_min_silence_ms,
  get_vad_trailing_filler_ms,
)

TRAILING_FILLERS = re.compile(
  r"\b(and|but|so|um+|uh+|like|well)\s*$",
  re.IGNORECASE,
)
INCOMPLETE_END = re.compile(r"[,\-:]$")


class AdaptiveEndpointTracker:
  """
  Tracks partial transcript text and suggests extra wait after fillers.
  Silero VAD still drives endpoint; this adds grace after last partial update.
  """

  def __init__(self) -> None:
    self._last_partial = ""

  def update_partial(self, text: str) -> None:
    self._last_partial = (text or "").strip()

  def reset(self) -> None:
    self._last_partial = ""

  def extra_silence_ms_after_speech_end(self) -> int:
    if not get_vad_adaptive_enabled():
      return 0
    t = self._last_partial
    if not t:
      return 0
    filler_ms = get_vad_trailing_filler_ms()
    if TRAILING_FILLERS.search(t):
      return filler_ms
    if INCOMPLETE_END.search(t):
      return filler_ms // 2
    return 0

  def effective_min_silence_ms(self) -> int:
    base = get_vad_min_silence_ms()
    if not get_vad_adaptive_enabled():
      return base
    t = self._last_partial
    if t and (TRAILING_FILLERS.search(t) or INCOMPLETE_END.search(t)):
      return base + get_vad_trailing_filler_ms()
    if t and t.endswith((".", "?", "!")):
      return max(250, base - 150)
    return base