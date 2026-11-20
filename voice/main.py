"""Standalone voice input: record audio, transcribe, confirm/edit text."""

from __future__ import annotations

import argparse
import os
import platform
import sys
import threading
import uuid

from voice.audio_devices import format_device_list
from voice.audio_prep import prepare_for_stt
from voice.config import (
    get_stt_model_size,
    set_input_device_override,
    set_stt_model_size_override,
)
from voice.endpointing import AdaptiveEndpointTracker
from voice.events import VoiceEventBus, default_event_bus
from voice.logger import VoiceLogger
from voice.recorder import capture_utterance
from voice.stt import preload_stt_model, transcribe_with_metadata, warmup_stt_model
from voice.telemetry import TurnTelemetry, duration_ms
from voice.transcript_confirm import (
    APPEND,
    RERECORD,
    TranscriptConfidence,
    prompt_transcript_edit,
)

if platform.system() == "Windows":
    try:
        import msvcrt
    except ImportError:
        msvcrt = None
else:
    msvcrt = None

_CLI_LOGGER = VoiceLogger(source="voice_cli")
_SESSION_ID: str | None = None
_STT_MODEL_LOGGED = False


def _missing_deps_message() -> str | None:
    missing = []
    for name, mod in [
        ("sounddevice", "sounddevice"),
        ("numpy", "numpy"),
        ("faster-whisper", "faster_whisper"),
        ("silero-vad", "silero_vad"),
    ]:
        try:
            __import__(mod)
        except ImportError:
            missing.append(name)
    if not missing:
        return None
    return (
        "Missing voice dependencies: "
        + ", ".join(missing)
        + "\nInstall with: pip install -r voice/requirements-stt.txt"
    )


def _ensure_session_logged(stt_model: str | None) -> None:
    global _SESSION_ID
    if _SESSION_ID is not None:
        return
    _SESSION_ID = str(uuid.uuid4())
    _CLI_LOGGER.log_session_start(
        _SESSION_ID,
        pid=os.getpid(),
        mode="cli",
        stt_model=stt_model or get_stt_model_size(),
    )


def _log_stt_model_ready_once(load_info) -> None:
    global _STT_MODEL_LOGGED
    if load_info is None or _STT_MODEL_LOGGED or _SESSION_ID is None:
        return
    _STT_MODEL_LOGGED = True
    _CLI_LOGGER.log_stt_model_ready(
        _SESSION_ID,
        stt_model=load_info.stt_model,
        stt_model_load_ms=round(load_info.stt_model_load_ms, 2),
        compute_type=load_info.compute_type,
        device=load_info.device,
    )


def _record_with_optional_ctrl_r(manual_stop: threading.Event) -> None:
    """On Windows native console, Ctrl+R sets manual_stop while recording."""
    if not msvcrt:
        return
    while not manual_stop.is_set():
        if msvcrt.kbhit():
            char = msvcrt.getch()
            if char == b"\x12":  # Ctrl+R
                manual_stop.set()
                print("\n[VOICE] Stopping recording (Ctrl+R)...", flush=True)
                break
        threading.Event().wait(0.05)


def _record_with_enter_stop(manual_stop: threading.Event) -> None:
    """Universal cross-platform stop: pressing Enter (blank line) sets manual_stop.

    Works in VSCode integrated terminal, Git Bash, and all POSIX shells where
    msvcrt is unavailable or unreliable.
    """
    try:
        sys.stdin.readline()
    except Exception:
        return
    if not manual_stop.is_set():
        manual_stop.set()
        # The \n clears the spinner/volume line before the message.
        print("\n[VOICE] Stopping recording (Enter)...", flush=True)


def _show_recording_indicator(manual_stop: threading.Event) -> None:
    """Print a spinner on a single terminal line until manual_stop or done."""
    frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    idx = 0
    while not manual_stop.is_set():
        frame = frames[idx % len(frames)]
        print(f"\r[● Recording... {frame}  Press Enter to stop ]  ", end="", flush=True)
        idx += 1
        threading.Event().wait(0.1)
    # Clear the indicator line
    print("\r" + " " * 55 + "\r", end="", flush=True)


class CaptureResult:
    __slots__ = (
        "raw_text",
        "telemetry",
        "audio_samples",
        "trimmed_samples",
        "audio_duration_sec",
        "stt_result",
        "load_info",
    )

    def __init__(
        self,
        raw_text: str | None,
        telemetry: TurnTelemetry,
        audio_samples: int | None,
        trimmed_samples: int | None,
        audio_duration_sec: float | None,
        stt_result=None,
        load_info=None,
    ) -> None:
        self.raw_text = raw_text
        self.telemetry = telemetry
        self.audio_samples = audio_samples
        self.trimmed_samples = trimmed_samples
        self.audio_duration_sec = audio_duration_sec
        self.stt_result = stt_result
        self.load_info = load_info


def _print_partial(prefix: str, text: str) -> None:
    line = text.replace("\n", " ").strip()
    if not line:
        return
    print(f"\r[partial] {line[:120]}{'…' if len(line) > 120 else ''}          ", flush=True)


def capture_and_transcribe(
    stt_model: str | None,
    *,
    input_device: int | None = None,
    event_bus: VoiceEventBus | None = None,
) -> CaptureResult:
    """Record from mic and return raw transcript (before user edit)."""
    set_stt_model_size_override(stt_model)
    bus = event_bus or default_event_bus()
    endpoint = AdaptiveEndpointTracker()

    manual_stop = threading.Event()
    cancel = threading.Event()
    if msvcrt:
        threading.Thread(
            target=_record_with_optional_ctrl_r,
            args=(manual_stop,),
            daemon=True,
        ).start()

    # Cross-platform Enter-key stop (works in VSCode, Git Bash, etc.)
    threading.Thread(
        target=_record_with_enter_stop,
        args=(manual_stop,),
        daemon=True,
    ).start()

    # Spinner indicator so the user knows the mic is active
    threading.Thread(
        target=_show_recording_indicator,
        args=(manual_stop,),
        daemon=True,
    ).start()

    print("[VOICE] Recording... (press Enter or Ctrl+R to stop)", flush=True)
    telemetry = TurnTelemetry()
    telemetry.mark_record_start()

    def on_partial(text: str) -> None:
        _print_partial("partial", text)
        bus.emit_partial(text)

    audio = capture_utterance(
        manual_stop=manual_stop,
        cancel=cancel,
        device=input_device,
        on_partial=on_partial,
        endpoint_tracker=endpoint,
    )
    # Ensure spinner and Enter-reader threads see the stop signal
    manual_stop.set()
    telemetry.mark_record_end()

    audio_samples = len(audio) if audio is not None else None

    if audio is None:
        return CaptureResult(
            None,
            telemetry,
            audio_samples,
            None,
            None,
        )

    prepared, trimmed_samples, duration_sec = prepare_for_stt(audio)

    print("[VOICE] Transcribing...", flush=True)
    stt_result, load_info = transcribe_with_metadata(prepared)
    telemetry.mark_stt_end()
    _log_stt_model_ready_once(load_info)

    text = stt_result.text if stt_result.text else None
    if text:
        bus.emit_final(text)
    return CaptureResult(
        text,
        telemetry,
        audio_samples,
        trimmed_samples,
        duration_sec,
        stt_result,
        load_info,
    )


def run_voice_input(
    stt_model: str | None,
    voice_logger: VoiceLogger | None = None,
    *,
    input_device: int | None = None,
    event_bus: VoiceEventBus | None = None,
) -> str | None:
    """
    Full voice-input flow: record → STT → show transcript → user edit → confirmed text.
    """
    vlog = voice_logger or _CLI_LOGGER
    turn_id = str(uuid.uuid4())
    model_name = stt_model or get_stt_model_size()
    _ensure_session_logged(stt_model)

    vlog.log_turn_start(turn_id, mode="cli", stt_model=model_name)

    bus = event_bus or default_event_bus()

    try:
        cap = capture_and_transcribe(
            stt_model,
            input_device=input_device,
            event_bus=bus,
        )
        telemetry = cap.telemetry
        record_ms = duration_ms(telemetry.record_started, telemetry.record_ended)

        if cap.raw_text is None and cap.audio_samples is None:
            vlog.log_record_end(turn_id, outcome="no_speech", record_ms=record_ms)
            vlog.log_turn_summary(
                turn_id,
                outcome="no_speech",
                metrics=telemetry.to_metrics_dict(),
            )
            print("[VOICE] No transcript (cancelled, no speech, or STT empty).")
            return None

        vlog.log_record_end(
            turn_id,
            outcome="ok",
            record_ms=record_ms,
            audio_samples=cap.audio_samples,
            audio_duration_sec=cap.audio_duration_sec,
            trimmed_samples=cap.trimmed_samples,
        )

        if cap.raw_text is None:
            stt_ms = duration_ms(telemetry.record_ended, telemetry.stt_ended)
            vlog.log_stt_result(
                turn_id,
                outcome="empty",
                stt_ms=stt_ms,
                stt_model=model_name,
                audio_duration_sec=cap.audio_duration_sec,
            )
            telemetry.mark_turn_end()
            vlog.log_turn_summary(
                turn_id,
                outcome="stt_empty",
                metrics=telemetry.to_metrics_dict(),
            )
            print("[VOICE] No transcript (cancelled, no speech, or STT empty).")
            return None

        stt_ms = duration_ms(telemetry.record_ended, telemetry.stt_ended)
        stt_meta = cap.stt_result
        vlog.log_stt_result(
            turn_id,
            outcome="ok",
            transcript_raw=cap.raw_text,
            stt_ms=stt_ms,
            stt_model=model_name,
            stt_beam_size=stt_meta.stt_beam_size if stt_meta else None,
            audio_duration_sec=cap.audio_duration_sec,
            language_probability=stt_meta.language_probability if stt_meta else None,
            avg_no_speech_prob=stt_meta.avg_no_speech_prob if stt_meta else None,
        )

        telemetry.mark_confirm_start()
        confidence = TranscriptConfidence(
            avg_logprob=stt_meta.avg_logprob if stt_meta else None,
            avg_no_speech_prob=stt_meta.avg_no_speech_prob if stt_meta else None,
            language_probability=stt_meta.language_probability if stt_meta else None,
        )
        confirmed = prompt_transcript_edit(cap.raw_text, confidence=confidence)
        telemetry.mark_confirm_end()
        telemetry.mark_turn_end()
        _print_telemetry(telemetry)

        if confirmed == RERECORD:
            print("[VOICE] Re-recording...")
            return run_voice_input(
                stt_model,
                vlog,
                input_device=input_device,
                event_bus=bus,
            )
        if confirmed == APPEND:
            print("[VOICE] Append another phrase...")
            extra = run_voice_input(
                stt_model,
                vlog,
                input_device=input_device,
                event_bus=bus,
            )
            if extra:
                merged = f"{cap.raw_text.strip()} {extra.strip()}".strip()
                print(f"\n[CONFIRMED INPUT] {merged}")
                bus.emit_confirmed(merged)
                return merged
            return None

        if confirmed is None:
            vlog.log_confirm(
                turn_id,
                outcome="discarded",
                transcript_raw=cap.raw_text,
            )
            vlog.log_turn_summary(
                turn_id,
                outcome="discarded",
                metrics=telemetry.to_metrics_dict(),
            )
            print("[VOICE] Discarded.")
            return None

        was_edited = confirmed.strip() != cap.raw_text.strip()
        vlog.log_confirm(
            turn_id,
            outcome="confirmed",
            transcript_raw=cap.raw_text,
            confirmed_text=confirmed,
            was_edited=was_edited,
        )
        vlog.log_turn_summary(
            turn_id,
            outcome="confirmed",
            metrics=telemetry.to_metrics_dict(),
        )
        print(f"\n[CONFIRMED INPUT] {confirmed}")
        bus.emit_confirmed(confirmed)
        return confirmed

    except Exception as exc:
        vlog.log_error(turn_id, stage="voice_input", error=str(exc))
        vlog.log_turn_summary(turn_id, outcome="error", metrics={})
        raise


def _print_telemetry(telemetry: TurnTelemetry) -> None:
    lines = telemetry.summary_lines()
    if lines:
        print("[VOICE] Timings:")
        for line in lines:
            print(line)


def _start_stt_preload(stt_model: str | None, *, warmup: bool) -> None:
    """Load Whisper in a background thread so first .talk is fast."""

    def _worker() -> None:
        set_stt_model_size_override(stt_model)
        load_info = preload_stt_model(stt_model)
        _log_stt_model_ready_once(load_info)
        if warmup:
            warmup_stt_model(stt_model)

    threading.Thread(target=_worker, daemon=True, name="stt-preload").start()


def run_repl(args: argparse.Namespace) -> int:
    msg = _missing_deps_message()
    if msg:
        print(msg, file=sys.stderr)
        return 1

    if args.device is not None:
        set_input_device_override(args.device)

    _ensure_session_logged(args.stt_model)
    if not args.no_preload:
        print("[VOICE] Loading speech model in background...", flush=True)
        _start_stt_preload(args.stt_model, warmup=not args.no_warmup)

    print("=" * 60)
    print("Local Voice Input (STT + confirm/edit)")
    print("Record speech, review the transcript, then edit if needed.")
    if args.continuous:
        print("Mode: continuous listen (auto .talk after each turn)")
    else:
        print("Commands: .talk  |  .devices  |  exit")
    print(f"STT model: {args.stt_model or get_stt_model_size()}")
    print(f"Audit log: {_CLI_LOGGER.log_path}")
    print("=" * 60)

    while True:
        if args.continuous:
            run_voice_input(args.stt_model, _CLI_LOGGER, input_device=args.device)
            continue
        try:
            line = input("\n[VOICE] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return 0

        if not line:
            continue
        lower = line.lower()
        if lower in {"exit", "quit"}:
            print("Goodbye.")
            return 0
        if lower == ".devices":
            print(format_device_list())
            continue
        if lower == ".talk":
            run_voice_input(args.stt_model, _CLI_LOGGER, input_device=args.device)
            continue

        print("Unknown command. Use .talk to record, .devices, or exit to quit.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local voice input: mic to confirmed text")
    parser.add_argument(
        "--stt-model",
        default=None,
        help="Whisper size override (base.en, small.en, distil-small.en, ...)",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="sounddevice input index (see --list-devices)",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List microphone devices and exit",
    )
    parser.add_argument(
        "--no-preload",
        action="store_true",
        help="Do not load STT model until first recording",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip warm-up decode after preload",
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Loop: record and confirm continuously until Ctrl+C",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Single .talk cycle then exit (still prompts for edit unless stdin closed)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_devices:
        print(format_device_list())
        return 0
    if args.once:
        _ensure_session_logged(args.stt_model)
        if args.device is not None:
            set_input_device_override(args.device)
        if not args.no_preload:
            preload_stt_model(args.stt_model)
            if not args.no_warmup:
                warmup_stt_model(args.stt_model)
        run_voice_input(args.stt_model, _CLI_LOGGER, input_device=args.device)
        return 0
    return run_repl(args)


if __name__ == "__main__":
    raise SystemExit(main())
