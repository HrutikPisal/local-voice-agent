"""Let the user review and edit a speech transcript before it is used."""


def format_transcript_preview(transcript: str) -> str:
    return transcript.strip()


def prompt_transcript_edit(transcript: str, input_fn=input) -> str | None:
    """
    Show what STT heard and let the user confirm or correct it.

    Returns confirmed text, or None if the user discards the turn.
    """
    cleaned = format_transcript_preview(transcript)
    if not cleaned:
        return None

    bar = "─" * 60
    print()
    print(bar)
    print("Voice transcript (this is what we heard):")
    print(f"  {cleaned}")
    print(bar)
    print("Press Enter to accept as-is, type corrections, or enter .cancel to discard.")
    try:
        edited = input_fn("Confirm or edit> ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None

    if edited == ".cancel":
        return None
    if not edited:
        return cleaned
    return edited
