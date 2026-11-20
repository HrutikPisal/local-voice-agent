"""Let the user review and edit a speech transcript before it is used."""

try:
    from prompt_toolkit import prompt as _pt_prompt
    _HAVE_PROMPT_TOOLKIT = True
except ImportError:
    _HAVE_PROMPT_TOOLKIT = False


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

    # When a custom input_fn is supplied (e.g. in tests), skip prompt_toolkit
    # and use it directly — this keeps existing tests passing.
    use_prompt_toolkit = _HAVE_PROMPT_TOOLKIT and input_fn is input

    if use_prompt_toolkit:
        print("Edit the transcript below, then press Enter to confirm (or type .cancel to discard).")
        try:
            edited = _pt_prompt("Confirm or edit> ", default=cleaned).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return None
    else:
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
