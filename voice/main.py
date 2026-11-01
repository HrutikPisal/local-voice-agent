"""Standalone voice input: record audio, transcribe, confirm/edit text."""

from __future__ import annotations

import argparse
import os
import platform
import sys
import threading
import uuid

from voice.audio_prep import prepare_for_stt
from voice.config import get_stt_model_size, set_stt_model_size_override
from voice.logger import VoiceLogger
from voice.recorder import capture_utterance
from voice.stt import transcribe_with_metadata
from voice.telemetry import TurnTelemetry, duration_ms
from voice.transcript_confirm import prompt_transcript_edit

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


def capture_and_transcribe(stt_model: str | None) -> CaptureResult:
    """Record from mic and return raw transcript (before user edit)."""
    set_stt_model_size_override(stt_model)

    manual_stop = threading.Event()
    cancel = threading.Event()
    if msvcrt:
        threading.Thread(
            target=_record_with_optional_ctrl_r,
            args=(manual_stop,),
            daemon=True,
        ).start()

    print("[VOICE] Recording... (Ctrl+R to stop on native Windows console)", flush=True)
    telemetry = TurnTelemetry()
    telemetry.mark_record_start()
    audio = capture_utterance(manual_stop=manual_stop, cancel=cancel)
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
) -> str | None:
    """
    Full voice-input flow: record → STT → show transcript → user edit → confirmed text.
    """
    vlog = voice_logger or _CLI_LOGGER
    turn_id = str(uuid.uuid4())
    model_name = stt_model or get_stt_model_size()
    _ensure_session_logged(stt_model)

    vlog.log_turn_start(turn_id, mode="cli", stt_model=model_name)

    try:
        cap = capture_and_transcribe(stt_model)
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
        confirmed = prompt_transcript_edit(cap.raw_text)
        telemetry.mark_confirm_end()
        telemetry.mark_turn_end()
        _print_telemetry(telemetry)

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


def run_repl(args: argparse.Namespace) -> int:
    msg = _missing_deps_message()
    if msg:
        print(msg, file=sys.stderr)
        return 1

    _ensure_session_logged(args.stt_model)

    print("=" * 60)
    print("Local Voice Input (STT + confirm/edit)")
    print("Record speech, review the transcript, then edit if needed.")
    print("Commands: .talk  |  exit")
    print(f"Audit log: {_CLI_LOGGER.log_path}")
    print("=" * 60)

    while True:
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
        if lower == ".talk":
            run_voice_input(args.stt_model, _CLI_LOGGER)
            continue

        print("Unknown command. Use .talk to record, or exit to quit.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local voice input: mic to confirmed text")
    parser.add_argument("--stt-model", default=None, help="Whisper size override (tiny, base, ...)")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Single .talk cycle then exit (still prompts for edit unless stdin closed)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.once:
        _ensure_session_logged(args.stt_model)
        run_voice_input(args.stt_model, _CLI_LOGGER)
        return 0
    return run_repl(args)


if __name__ == "__main__":
    raise SystemExit(main())
