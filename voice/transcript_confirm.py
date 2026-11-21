"""Let the user review and edit a speech transcript before it is used."""

from __future__ import annotations

import os
from dataclasses import dataclass

from voice.config import (
    get_confirm_auto_accept_enabled,
    get_confirm_auto_accept_max_no_speech,
    get_confirm_auto_accept_min_logprob,
)

try:
    from prompt_toolkit import prompt as _pt_prompt

    _HAVE_PROMPT_TOOLKIT = True
except ImportError:
    _HAVE_PROMPT_TOOLKIT = False

RERECORD = "__VOICE_RERECORD__"
APPEND = "__VOICE_APPEND__"

# Set from CLI (--plain-confirm / --rich-confirm); None = auto-detect terminal.
_FORCE_PLAIN_CONFIRM: bool | None = None


def set_force_plain_confirm(value: bool | None) -> None:
    """Override plain vs prompt-toolkit confirm (None restores auto-detect)."""
    global _FORCE_PLAIN_CONFIRM
    _FORCE_PLAIN_CONFIRM = value


def should_use_plain_confirm() -> bool:
    """Use plain input() when integrated terminals break prompt-toolkit editing."""
    if _FORCE_PLAIN_CONFIRM is not None:
        return _FORCE_PLAIN_CONFIRM
    if os.environ.get("VOICE_PLAIN_CONFIRM", "").strip().lower() in {"1", "true", "yes"}:
        return True
    term = (os.environ.get("TERM_PROGRAM") or "").strip().lower()
    if term in {"vscode", "cursor"}:
        return True
    return False


@dataclass
class TranscriptConfidence:
    avg_logprob: float | None = None
    avg_no_speech_prob: float | None = None
    language_probability: float | None = None


def format_transcript_preview(transcript: str) -> str:
    return transcript.strip()


def should_auto_accept(confidence: TranscriptConfidence | None) -> bool:
    if not get_confirm_auto_accept_enabled():
        return False
    if confidence is None:
        return False
    if confidence.avg_logprob is not None and confidence.avg_logprob < get_confirm_auto_accept_min_logprob():
        return False
    if confidence.avg_no_speech_prob is not None:
        if confidence.avg_no_speech_prob > get_confirm_auto_accept_max_no_speech():
            return False
    if confidence.language_probability is not None and confidence.language_probability < 0.5:
        return False
    return True


def _resolve_edit_input(
    cleaned: str,
    input_fn,
    *,
    plain_confirm: bool,
) -> str:
    use_prompt_toolkit = _HAVE_PROMPT_TOOLKIT and input_fn is input and not plain_confirm

    if use_prompt_toolkit:
        print(
            "Edit below, Enter to confirm, .cancel to discard, .rerecord / .append for more audio."
        )
        return _pt_prompt("Confirm or edit> ", default=cleaned).strip()

    print("Press Enter to accept as-is, or type the full corrected line.")
    print("Commands: .cancel  .rerecord  .append")
    return input_fn("Confirm or edit> ").strip()


def prompt_transcript_edit(
    transcript: str,
    input_fn=input,
    confidence: TranscriptConfidence | None = None,
    *,
    allow_auto: bool = True,
    plain_confirm: bool | None = None,
) -> str | None:
    """
    Show what STT heard and let the user confirm or correct it.

    Returns confirmed text, None if discarded, RERECORD to capture again, or APPEND to add speech.
    """
    cleaned = format_transcript_preview(transcript)
    if not cleaned:
        return None

    if allow_auto and should_auto_accept(confidence):
        print()
        print("[VOICE] High-confidence transcript — auto-accepted (confirm_auto_accept_enabled).")
        print(f"  {cleaned}")
        return cleaned

    bar = "─" * 60
    print()
    print(bar)
    print("Voice transcript (this is what we heard):")
    print(f"  {cleaned}")
    print(bar)

    use_plain = plain_confirm if plain_confirm is not None else should_use_plain_confirm()
    if use_plain and input_fn is input:
        print("(Plain confirm — empty line editor; type the full line or press Enter to accept.)")

    try:
        edited = _resolve_edit_input(cleaned, input_fn, plain_confirm=use_plain)
    except (EOFError, KeyboardInterrupt):
        print()
        return None

    low = edited.lower()
    if low == ".cancel":
        return None
    if low == ".rerecord":
        return RERECORD
    if low == ".append":
        return APPEND
    if not edited:
        return cleaned
    return edited