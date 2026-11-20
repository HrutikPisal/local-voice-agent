"""Domain hotwords and post-STT replacements."""

from __future__ import annotations

import json
import re
from pathlib import Path

from voice.config import REPO_ROOT

DEFAULT_HOTWORDS_PATH = REPO_ROOT / "config" / "hotwords.json"

_cache: dict | None = None


def _load_hotwords() -> dict:
  global _cache
  if _cache is not None:
    return _cache
  path = DEFAULT_HOTWORDS_PATH
  if not path.exists():
    _cache = {"replacements": {}, "hint_terms": []}
    return _cache
  with open(path, encoding="utf-8") as handle:
    data = json.load(handle)
  _cache = {
    "replacements": data.get("replacements") or {},
    "hint_terms": data.get("hint_terms") or [],
  }
  return _cache


def apply_hotword_replacements(text: str) -> str:
  if not text:
    return text
  reps = _load_hotwords().get("replacements") or {}
  out = text
  for wrong, right in reps.items():
    if not wrong:
      continue
    pattern = re.compile(re.escape(wrong), re.IGNORECASE)
    out = pattern.sub(right, out)
  return out


def hotword_hint_suffix() -> str:
  terms = _load_hotwords().get("hint_terms") or []
  if not terms:
    return ""
  return " Terms: " + ", ".join(terms[:24]) + "."