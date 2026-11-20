"""Pluggable STT engines (CPU-first). Default: faster-whisper."""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass
from typing import Any

from voice.config import (
    get_stt_beam_size,
    get_stt_compute_type,
    get_stt_condition_on_previous_text,
    get_stt_initial_prompt,
    get_stt_language,
    get_stt_model_size,
    get_stt_vad_filter,
)


@dataclass
class EngineDecodeResult:
    text: str
    language_probability: float | None = None
    avg_no_speech_prob: float | None = None
    avg_logprob: float | None = None


@dataclass
class EngineLoadInfo:
    stt_model_load_ms: float
    stt_model: str
    compute_type: str
    device: str
    engine_name: str = "faster_whisper"


class STTEngine(abc.ABC):
    """Abstract speech-to-text engine."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        ...

    @abc.abstractmethod
    def load(self) -> EngineLoadInfo | None:
        """Load models if needed; return load timing on first load."""

    @abc.abstractmethod
    def transcribe(
        self,
        audio,
        *,
        word_timestamps: bool = False,
        beam_size: int | None = None,
    ) -> EngineDecodeResult:
        ...


class FasterWhisperEngine(STTEngine):
    """faster-whisper (CTranslate2) backend."""

    def __init__(
        self,
        model_size: str | None = None,
        compute_type: str | None = None,
        device: str = "cpu",
        cpu_threads: int | None = None,
        num_workers: int = 1,
    ) -> None:
        self._model_size = model_size or get_stt_model_size()
        self._compute_type = compute_type or get_stt_compute_type()
        self._device = device
        self._cpu_threads = cpu_threads
        self._num_workers = num_workers
        self._model: Any = None
        self._loaded = False

    @property
    def name(self) -> str:
        return "faster_whisper"

    def _model_kwargs(self) -> dict[str, Any]:
        kw: dict[str, Any] = {
            "device": self._device,
            "compute_type": self._compute_type,
            "num_workers": self._num_workers,
        }
        if self._cpu_threads is not None:
            kw["cpu_threads"] = self._cpu_threads
        return kw

    def load(self) -> EngineLoadInfo | None:
        if self._loaded:
            return None
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise ImportError(
                "faster-whisper is required. Install with: pip install -r voice/requirements-stt.txt"
            ) from exc
        started = time.perf_counter()
        self._model = WhisperModel(self._model_size, **self._model_kwargs())
        load_ms = (time.perf_counter() - started) * 1000
        self._loaded = True
        return EngineLoadInfo(
            stt_model_load_ms=load_ms,
            stt_model=self._model_size,
            compute_type=self._compute_type,
            device=self._device,
            engine_name=self.name,
        )

    def transcribe(
        self,
        audio,
        *,
        word_timestamps: bool = False,
        beam_size: int | None = None,
    ) -> EngineDecodeResult:
        if not self._loaded:
            self.load()
        assert self._model is not None

        from voice.hotwords import hotword_hint_suffix

        initial_prompt = get_stt_initial_prompt()
        hint = hotword_hint_suffix()
        if hint and initial_prompt:
            initial_prompt = initial_prompt + hint
        elif hint:
            initial_prompt = hint.strip()
        prompt_kw = {"initial_prompt": initial_prompt} if initial_prompt else {}
        bs = beam_size if beam_size is not None else get_stt_beam_size()

        segments, info = self._model.transcribe(
            audio,
            language=get_stt_language(),
            beam_size=bs,
            vad_filter=get_stt_vad_filter(),
            condition_on_previous_text=get_stt_condition_on_previous_text(),
            word_timestamps=word_timestamps,
            **prompt_kw,
        )
        seg_list = list(segments)
        parts = [seg.text.strip() for seg in seg_list if seg.text.strip()]
        text = " ".join(parts).strip()

        no_speech_probs = [
            seg.no_speech_prob
            for seg in seg_list
            if getattr(seg, "no_speech_prob", None) is not None
        ]
        avg_no_speech = sum(no_speech_probs) / len(no_speech_probs) if no_speech_probs else None
        logprobs = [
            seg.avg_logprob for seg in seg_list if getattr(seg, "avg_logprob", None) is not None
        ]
        avg_logprob = sum(logprobs) / len(logprobs) if logprobs else None
        lang_prob = getattr(info, "language_probability", None)

        return EngineDecodeResult(
            text=text,
            language_probability=lang_prob,
            avg_no_speech_prob=avg_no_speech,
            avg_logprob=avg_logprob,
        )


class SherpaOnnxEngine(STTEngine):
    """Optional sherpa-onnx engine (install sherpa-onnx separately)."""

    def __init__(self, model_dir: str | None = None) -> None:
        self._model_dir = model_dir
        self._recognizer: Any = None
        self._loaded = False

    @property
    def name(self) -> str:
        return "sherpa_onnx"

    def load(self) -> EngineLoadInfo | None:
        if self._loaded:
            return None
        try:
            import sherpa_onnx
        except ImportError as exc:
            raise ImportError(
                "sherpa-onnx is optional. pip install sherpa-onnx and set stt_engine to sherpa_onnx."
            ) from exc
        if not self._model_dir:
            raise ValueError("sherpa_onnx requires stt_sherpa_model_dir in config.json")
        started = time.perf_counter()
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            tokens=f"{self._model_dir}/tokens.txt",
            encoder=f"{self._model_dir}/encoder-epoch-99-avg-1.onnx",
            decoder=f"{self._model_dir}/decoder-epoch-99-avg-1.onnx",
            joiner=f"{self._model_dir}/joiner-epoch-99-avg-1.onnx",
            num_threads=4,
        )
        load_ms = (time.perf_counter() - started) * 1000
        self._loaded = True
        return EngineLoadInfo(
            stt_model_load_ms=load_ms,
            stt_model=self._model_dir,
            compute_type="onnx",
            device="cpu",
            engine_name=self.name,
        )

    def transcribe(
        self,
        audio,
        *,
        word_timestamps: bool = False,
        beam_size: int | None = None,
    ) -> EngineDecodeResult:
        if not self._loaded:
            self.load()
        import numpy as np

        stream = self._recognizer.create_stream()
        samples = np.asarray(audio, dtype=np.float32)
        stream.accept_waveform(16000, samples)
        self._recognizer.decode_stream(stream)
        text = stream.result.text.strip()
        return EngineDecodeResult(text=text)


class MoonshineOnnxEngine(STTEngine):
    """Optional Moonshine ONNX (install moonshine-onnx separately)."""

    def __init__(self, variant: str = "tiny") -> None:
        self._variant = variant
        self._model: Any = None
        self._loaded = False

    @property
    def name(self) -> str:
        return "moonshine_onnx"

    def load(self) -> EngineLoadInfo | None:
        if self._loaded:
            return None
        try:
            from moonshine_onnx import MoonshineOnnxModel
        except ImportError as exc:
            raise ImportError(
                "moonshine-onnx is optional. pip install moonshine-onnx for this engine."
            ) from exc
        started = time.perf_counter()
        self._model = MoonshineOnnxModel(model_name=self._variant)
        load_ms = (time.perf_counter() - started) * 1000
        self._loaded = True
        return EngineLoadInfo(
            stt_model_load_ms=load_ms,
            stt_model=self._variant,
            compute_type="onnx",
            device="cpu",
            engine_name=self.name,
        )

    def transcribe(
        self,
        audio,
        *,
        word_timestamps: bool = False,
        beam_size: int | None = None,
    ) -> EngineDecodeResult:
        if not self._loaded:
            self.load()
        text = self._model.generate(audio)[0].strip()
        return EngineDecodeResult(text=text)


_engines: dict[tuple, STTEngine] = {}


def create_stt_engine(
    engine_name: str | None = None,
    model_size: str | None = None,
    **kwargs: Any,
) -> STTEngine:
    from voice.config import (
        get_stt_cpu_threads,
        get_stt_engine,
        get_stt_num_workers,
        get_stt_sherpa_model_dir,
    )

    name = (engine_name or get_stt_engine()).lower()
    size = model_size or get_stt_model_size()
    cache_key = (name, size, kwargs.get("device", "cpu"))

    if cache_key in _engines:
        return _engines[cache_key]

    if name in {"faster_whisper", "whisper", "default"}:
        engine: STTEngine = FasterWhisperEngine(
            model_size=size,
            cpu_threads=get_stt_cpu_threads(),
            num_workers=get_stt_num_workers(),
            **{k: v for k, v in kwargs.items() if k in {"compute_type", "device"}},
        )
    elif name == "sherpa_onnx":
        engine = SherpaOnnxEngine(model_dir=get_stt_sherpa_model_dir())
    elif name in {"moonshine", "moonshine_onnx"}:
        engine = MoonshineOnnxEngine(variant=size if size in {"tiny", "base"} else "tiny")
    else:
        raise ValueError(f"Unknown stt_engine: {name}")

    _engines[cache_key] = engine
    return engine


def preload_engine(engine: STTEngine | None = None) -> EngineLoadInfo | None:
    eng = engine or create_stt_engine()
    return eng.load()


def warmup_engine(engine: STTEngine | None = None, sample_rate: int = 16000) -> None:
    import numpy as np

    eng = engine or create_stt_engine()
    eng.load()
    silence = np.zeros(int(0.25 * sample_rate), dtype=np.float32)
    eng.transcribe(silence, beam_size=1)


def clear_engine_cache() -> None:
    global _engines
    _engines.clear()