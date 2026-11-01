"""
Self-hosted LiveKit agent: local STT only.

User speaks in a LiveKit room (Agents Playground or your client). The worker
transcribes with faster-whisper and prints the transcript for confirm/edit.
No cloud LiveKit Inference — you control the server and models.

Run (after docker compose up):
  pip install -r voice/requirements-livekit.txt
  copy voice/livekit/.env.example voice/livekit/.env.local
  python -m voice.livekit console
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import textwrap
import uuid
from pathlib import Path

from dotenv import load_dotenv
from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli, room_io
from livekit.plugins import openai as lk_openai
from livekit.plugins import silero

from voice.config import get_default_ollama_model, get_stt_model_size
from voice.livekit.stt_faster_whisper import FasterWhisperSTT
from voice.logger import VoiceLogger
from voice.telemetry import TurnTelemetry, duration_ms
from voice.transcript_confirm import prompt_transcript_edit

logger = logging.getLogger("voice.livekit.agent")

_ENV_PATH = Path(__file__).resolve().parent / ".env.local"
load_dotenv(_ENV_PATH)
load_dotenv()

_LK_LOGGER = VoiceLogger(source="voice_livekit")


class TranscriptReviewAgent(Agent):
    """
    STT-only UX: on each user turn, show transcript for edit/confirm.
    Does not call generate_reply() — no spoken/text assistant response.
    """

    def __init__(self, room, voice_logger: VoiceLogger) -> None:
        self._room = room
        self._vlog = voice_logger
        super().__init__(
            instructions=textwrap.dedent(
                """
                You are a speech-to-text listener. Do not generate spoken replies.
                Wait for the user's speech to be transcribed locally.
                """
            ),
            llm=lk_openai.LLM.with_ollama(
                model=get_default_ollama_model(),
                base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1"),
            ),
        )

    async def on_user_turn_completed(self, turn_ctx, new_message) -> None:
        raw = (new_message.text_content or "").strip()
        if not raw:
            return

        turn_id = str(uuid.uuid4())
        stt_model = os.getenv("VOICE_STT_MODEL") or get_stt_model_size()
        telemetry = TurnTelemetry()
        stt_ms = self._vlog_stt_ms_from_extra(new_message)

        self._vlog.log_turn_start(
            turn_id,
            mode="livekit",
            stt_model=stt_model,
            room=self._room.name,
        )
        self._vlog.log_stt_result(
            turn_id,
            outcome="ok",
            transcript_raw=raw,
            stt_ms=stt_ms,
            stt_model=stt_model,
        )

        _print_transcript(raw)
        telemetry.mark_confirm_start()
        confirmed = await asyncio.to_thread(prompt_transcript_edit, raw)
        telemetry.mark_confirm_end()
        telemetry.mark_turn_end()

        if confirmed is None:
            self._vlog.log_confirm(
                turn_id,
                outcome="discarded",
                transcript_raw=raw,
            )
            self._vlog.log_turn_summary(
                turn_id,
                outcome="discarded",
                metrics=_livekit_metrics(telemetry, stt_ms),
            )
            print("[VOICE] Discarded.")
            return

        was_edited = confirmed.strip() != raw.strip()
        self._vlog.log_confirm(
            turn_id,
            outcome="confirmed",
            transcript_raw=raw,
            confirmed_text=confirmed,
            was_edited=was_edited,
        )
        self._vlog.log_turn_summary(
            turn_id,
            outcome="confirmed",
            metrics=_livekit_metrics(telemetry, stt_ms),
        )
        print(f"\n[CONFIRMED INPUT] {confirmed}")
        await _publish_transcript(self._room, confirmed, turn_id, self._vlog)

    @staticmethod
    def _vlog_stt_ms_from_extra(new_message) -> float | None:
        meta = getattr(new_message, "metadata", None) or {}
        if isinstance(meta, dict):
            val = meta.get("stt_ms")
            if val is not None:
                return float(val)
        return None


def _livekit_metrics(
    telemetry: TurnTelemetry, stt_ms: float | None
) -> dict[str, float | None]:
    metrics = telemetry.to_metrics_dict()
    if stt_ms is not None:
        metrics["stt_ms"] = stt_ms
    return metrics


server = AgentServer()


def _print_transcript(raw: str) -> None:
    bar = "─" * 60
    print()
    print(bar)
    print("LiveKit voice transcript (local STT):")
    print(f"  {raw}")
    print(bar)
    print("Edit in this terminal: Enter = accept, type fixes, .cancel = discard")


async def _publish_transcript(
    room,
    text: str,
    turn_id: str,
    vlog: VoiceLogger,
) -> None:
    topic = "voice_stt"
    payload = json.dumps({"type": "voice_transcript", "text": text}).encode("utf-8")
    try:
        await room.local_participant.publish_data(payload, reliable=True, topic=topic)
        vlog.log_publish(turn_id, topic=topic, success=True)
    except Exception as exc:
        vlog.log_publish(turn_id, topic=topic, success=False, error=str(exc))
        logger.warning("Could not publish transcript data: %s", exc)


@server.rtc_session(agent_name="local-cli-voice-stt")
async def voice_stt_entrypoint(ctx: JobContext) -> None:
    ctx.log_context_fields = {"room": ctx.room.name}
    session_id = str(uuid.uuid4())
    _LK_LOGGER.log_session_start(
        session_id,
        pid=os.getpid(),
        mode="livekit",
        room=ctx.room.name,
        stt_model=os.getenv("VOICE_STT_MODEL") or get_stt_model_size(),
    )

    session = AgentSession(
        stt=FasterWhisperSTT(
            model_size=os.getenv("VOICE_STT_MODEL"),
            language=os.getenv("VOICE_STT_LANGUAGE"),
            compute_type=os.getenv("VOICE_STT_COMPUTE"),
        ),
        vad=silero.VAD.load(),
    )

    await session.start(
        agent=TranscriptReviewAgent(ctx.room, _LK_LOGGER),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(),
        ),
    )
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(server)
