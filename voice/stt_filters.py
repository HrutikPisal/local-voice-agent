"""Post-decode filters for Whisper hallucinations and noise."""

from __future__ import annotations

# Common Whisper phantom phrases on silence / noise (lowercase match).
HALLUCINATION_PHRASES = frozenset(
  {
    "thank you for watching",
    "thanks for watching",
    "subscribe",
    "like and subscribe",
    "you",
    "bye",
    "thank you",
    "thanks",
    "music",
    "applause",
    "silence",
    "subtitle",
    "subtitles",
    "www.",
    "http",
  }
)


def normalize_for_filter(text: str) -> str:
  return " ".join(text.lower().strip().split())


def is_likely_hallucination(text: str, avg_no_speech_prob: float | None) -> bool:
  if not text or not text.strip():
    return False
  norm = normalize_for_filter(text)
  if norm in HALLUCINATION_PHRASES:
    return True
  for phrase in HALLUCINATION_PHRASES:
    if len(phrase) > 8 and phrase in norm and len(norm.split()) <= 6:
      return True
  if avg_no_speech_prob is not None and avg_no_speech_prob > 0.65 and len(norm.split()) <= 2:
    return True
  return False


def apply_no_speech_filter(
  text: str,
  avg_no_speech_prob: float | None,
  *,
  max_words_with_high_no_speech: int = 3,
  no_speech_threshold: float = 0.7,
) -> str:
  if avg_no_speech_prob is not None and avg_no_speech_prob > no_speech_threshold:
    if len(text.split()) < max_words_with_high_no_speech:
      return ""
  if is_likely_hallucination(text, avg_no_speech_prob):
    return ""
  return text