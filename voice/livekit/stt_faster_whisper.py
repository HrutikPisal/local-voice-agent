"""LiveKit STT plugin backed by local faster-whisper (same engine as voice/stt.py)."""

from __future__ import annotations

import asyncio
import uuid

import numpy as np
from livekit.agents import stt
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer

from voice.audio_prep import prepare_for_stt
from voice.config import get_stt_compute_type, get_stt_language, get_stt_model_size
from voice.stt import transcribe_with_metadata


def _buffer_to_float32(buffer: AudioBuffer) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for frame in buffer:
        chunk = np.frombuffer(frame.data, dtype=np.int16)
        if frame.num_channels > 1:
            chunk = chunk.reshape(-1, frame.num_channels)[:, 0]
        chunks.append(chunk)
    if not chunks:
        return np.array([], dtype=np.float32)
    pcm = np.concatenate(chunks)
    return pcm.astype(np.float32) / 32768.0


class FasterWhisperSTT(stt.STT):
    """Batch STT using faster-whisper; pair with Silero VAD + StreamAdapter in the agent."""

    def __init__(
        self,
        *,
        model_size: str | None = None,
        language: str | None = None,
        compute_type: str | None = None,
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=False,
                interim_results=False,
            )
        )
        self._model_size = model_size or get_stt_model_size()
        self._language = language or get_stt_language()
        self._compute_type = compute_type or get_stt_compute_type()

    @property
    def model(self) -> str:
        return self._model_size

    @property
    def provider(self) -> str:
        return "faster-whisper-local"

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options=DEFAULT_API_CONNECT_OPTIONS,  # noqa: ARG002
    ) -> stt.SpeechEvent:
        audio = _buffer_to_float32(buffer)
        if len(audio) == 0:
            return stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                request_id=str(uuid.uuid4()),
                alternatives=[],
            )

        lang = self._language if language is NOT_GIVEN else language
        prepared, _trimmed, _duration = prepare_for_stt(audio)
        result, _load = await asyncio.to_thread(transcribe_with_metadata, prepared)
        text = result.text
        confidence = result.language_probability if result.language_probability is not None else (
            1.0 if text else 0.0
        )
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            request_id=str(uuid.uuid4()),
            alternatives=[
                stt.SpeechData(
                    language=lang,
                    text=text,
                    confidence=confidence,
                )
            ],
        )
