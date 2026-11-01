"""Simple latency markers for the voice loop."""

import time
from dataclasses import dataclass, field


def duration_ms(start: float | None, end: float | None) -> float | None:
    if start is None or end is None:
        return None
    return (end - start) * 1000


@dataclass
class TurnTelemetry:
    """Timestamps for one user turn (monotonic seconds)."""

    record_started: float | None = None
    record_ended: float | None = None
    stt_ended: float | None = None
    confirm_started: float | None = None
    confirm_ended: float | None = None
    llm_first_token: float | None = None
    tts_first_audio: float | None = None
    turn_ended: float | None = None
    extra: dict = field(default_factory=dict)

    def mark_record_start(self) -> None:
        self.record_started = time.monotonic()

    def mark_record_end(self) -> None:
        self.record_ended = time.monotonic()

    def mark_stt_end(self) -> None:
        self.stt_ended = time.monotonic()

    def mark_confirm_start(self) -> None:
        self.confirm_started = time.monotonic()

    def mark_confirm_end(self) -> None:
        self.confirm_ended = time.monotonic()

    def mark_llm_first_token(self) -> None:
        if self.llm_first_token is None:
            self.llm_first_token = time.monotonic()

    def mark_tts_first_audio(self) -> None:
        if self.tts_first_audio is None:
            self.tts_first_audio = time.monotonic()

    def mark_turn_end(self) -> None:
        self.turn_ended = time.monotonic()

    def to_metrics_dict(self) -> dict[str, float | None]:
        end = self.turn_ended or self.confirm_ended or self.stt_ended
        return {
            "record_ms": duration_ms(self.record_started, self.record_ended),
            "stt_ms": duration_ms(self.record_ended, self.stt_ended),
            "confirm_ms": duration_ms(self.confirm_started, self.confirm_ended),
            "total_ms": duration_ms(self.record_started, end),
        }

    def summary_lines(self) -> list[str]:
        metrics = self.to_metrics_dict()
        lines = []
        if metrics["record_ms"] is not None:
            lines.append(f"  record: {metrics['record_ms']:.0f} ms")
        if metrics["stt_ms"] is not None:
            lines.append(f"  stt: {metrics['stt_ms']:.0f} ms")
        if metrics["confirm_ms"] is not None:
            lines.append(f"  confirm: {metrics['confirm_ms']:.0f} ms")
        if metrics["total_ms"] is not None:
            lines.append(f"  total: {metrics['total_ms']:.0f} ms")
        if self.llm_first_token and self.stt_ended:
            ttft_ms = duration_ms(self.stt_ended, self.llm_first_token)
            if ttft_ms is not None:
                lines.append(f"  llm TTFT: {ttft_ms:.0f} ms")
        if self.tts_first_audio and self.record_ended:
            ttfa_ms = duration_ms(self.record_ended, self.tts_first_audio)
            if ttfa_ms is not None:
                lines.append(f"  TTFA (after speech end): {ttfa_ms:.0f} ms")
        return lines
