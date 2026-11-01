"""Structured voice activity logs (JSONL), parallel to the main CLI AgentLogger."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from voice.config import LOGS_DIR


class VoiceLogger:
    """Writes voice turn events to date-based log files under logs/."""

    def __init__(
        self,
        log_file: str | None = None,
        source: str = "voice_cli",
        log_path: Path | None = None,
    ) -> None:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        if log_path is not None:
            self.log_path = log_path
        elif log_file is None:
            date_str = datetime.now().strftime("%d_%m_%Y")
            log_file = f"voice_{date_str}.txt"
            self.log_path = LOGS_DIR / log_file
        else:
            self.log_path = LOGS_DIR / log_file
        self.source = source

    def _write(self, turn_id: str, entry: dict) -> None:
        entry["source"] = self.source
        entry["turn_id"] = turn_id
        entry["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = json.dumps(entry, ensure_ascii=False)
        with open(self.log_path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def log_session_start(self, turn_id: str, **fields: Any) -> None:
        payload = {"event": "voice_session_start", **fields}
        self._write(turn_id, payload)

    def log_turn_start(
        self,
        turn_id: str,
        *,
        mode: str,
        stt_model: str,
        room: str | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "event": "voice_turn_start",
            "mode": mode,
            "stt_model": stt_model,
        }
        if room:
            payload["room"] = room
        self._write(turn_id, payload)

    def log_stt_model_ready(self, session_id: str, **fields: Any) -> None:
        payload = {"event": "voice_stt_model_ready", **fields}
        self._write(session_id, payload)

    def log_record_end(
        self,
        turn_id: str,
        *,
        outcome: str,
        record_ms: float | None = None,
        audio_samples: int | None = None,
        audio_duration_sec: float | None = None,
        trimmed_samples: int | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "event": "voice_record_end",
            "outcome": outcome,
        }
        if record_ms is not None:
            payload["record_ms"] = round(record_ms, 2)
        if audio_samples is not None:
            payload["audio_samples"] = audio_samples
        if audio_duration_sec is not None:
            payload["audio_duration_sec"] = round(audio_duration_sec, 3)
        if trimmed_samples is not None:
            payload["trimmed_samples"] = trimmed_samples
        self._write(turn_id, payload)

    def log_stt_result(
        self,
        turn_id: str,
        *,
        outcome: str,
        transcript_raw: str = "",
        stt_ms: float | None = None,
        stt_model: str | None = None,
        stt_beam_size: int | None = None,
        audio_duration_sec: float | None = None,
        language_probability: float | None = None,
        avg_no_speech_prob: float | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "event": "voice_stt_result",
            "outcome": outcome,
            "transcript_raw": transcript_raw,
            "transcript_length": len(transcript_raw),
        }
        if stt_ms is not None:
            payload["stt_ms"] = round(stt_ms, 2)
        if stt_model:
            payload["stt_model"] = stt_model
        if stt_beam_size is not None:
            payload["stt_beam_size"] = stt_beam_size
        if audio_duration_sec is not None:
            payload["audio_duration_sec"] = round(audio_duration_sec, 3)
        if language_probability is not None:
            payload["language_probability"] = round(language_probability, 4)
        if avg_no_speech_prob is not None:
            payload["avg_no_speech_prob"] = round(avg_no_speech_prob, 4)
        self._write(turn_id, payload)

    def log_confirm(
        self,
        turn_id: str,
        *,
        outcome: str,
        transcript_raw: str,
        confirmed_text: str | None = None,
        was_edited: bool | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "event": "voice_confirm",
            "outcome": outcome,
            "transcript_raw": transcript_raw,
        }
        if confirmed_text is not None:
            payload["confirmed_text"] = confirmed_text
        if was_edited is not None:
            payload["was_edited"] = was_edited
        self._write(turn_id, payload)

    def log_publish(
        self,
        turn_id: str,
        *,
        topic: str,
        success: bool,
        error: str | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "event": "voice_publish",
            "topic": topic,
            "success": success,
        }
        if error:
            payload["error"] = error
        self._write(turn_id, payload)

    def log_turn_summary(
        self,
        turn_id: str,
        *,
        outcome: str,
        metrics: dict[str, float | None],
    ) -> None:
        payload: dict[str, Any] = {
            "event": "voice_turn_summary",
            "outcome": outcome,
        }
        for key, value in metrics.items():
            if value is not None:
                payload[key] = round(value, 2)
        self._write(turn_id, payload)

    def log_error(self, turn_id: str, *, stage: str, error: str) -> None:
        self._write(
            turn_id,
            {"event": "voice_error", "stage": stage, "error": error},
        )
