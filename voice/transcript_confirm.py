"""Let the user review and edit a speech transcript before it is used."""

from __future__ import annotations

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


def prompt_transcript_edit(
    transcript: str,
    input_fn=input,
    confidence: TranscriptConfidence | None = None,
    *,
    allow_auto: bool = True,
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

    use_prompt_toolkit = _HAVE_PROMPT_TOOLKIT and input_fn is input

    if use_prompt_toolkit:
        print(
            "Edit below, Enter to confirm, .cancel to discard, .rerecord / .append for more audio."
        )
        try:
            edited = _pt_prompt("Confirm or edit> ", default=cleaned).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return None
    else:
        print(
            "Press Enter to accept, type corrections, .cancel, .rerecord, or .append."
        )
        try:
            edited = input_fn("Confirm or edit> ").strip()
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