"""Prepare LLM text for TTS (strip non-speakable formatting)."""

import re

_EMOJI = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_MD_BOLD = re.compile(r"\*\*([^*]+)\*\*")
_MD_ITALIC = re.compile(r"\*([^*]+)\*")
_MD_CODE = re.compile(r"`([^`]+)`")
_MD_HEADING = re.compile(r"^#+\s*", re.MULTILINE)
_MD_BULLET = re.compile(r"^\s*[-*]\s+", re.MULTILINE)


def sanitize_for_tts(text: str) -> str:
    """Strip markdown and emoji so Piper can speak the result naturally."""
    if not text:
        return ""
    out = text.strip()
    out = _EMOJI.sub("", out)
    out = _MD_LINK.sub(r"\1", out)
    out = _MD_BOLD.sub(r"\1", out)
    out = _MD_ITALIC.sub(r"\1", out)
    out = _MD_CODE.sub(r"\1", out)
    out = _MD_HEADING.sub("", out)
    out = _MD_BULLET.sub("", out)
    out = re.sub(r"\s+", " ", out)
    return out.strip()
