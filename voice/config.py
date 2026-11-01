"""Load voice package settings without importing the main CLI config module."""

import json
from pathlib import Path

VOICE_DIR = Path(__file__).resolve().parent
REPO_ROOT = VOICE_DIR.parent
LOGS_DIR = REPO_ROOT / "logs"
MODELS_DIR = VOICE_DIR / "models" / "voices"
CONFIG_PATH = VOICE_DIR / "config.json"
OLLAMA_MODELS_PATH = REPO_ROOT / "config" / "models.json"


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


_VOICE = _load_json(CONFIG_PATH)


def get_voice_setting(key: str, default=None):
    return _VOICE.get(key, default)


def get_sample_rate() -> int:
    return int(get_voice_setting("sample_rate", 16000))


def get_vad_chunk_samples() -> int:
    return int(get_voice_setting("vad_chunk_samples", 512))


def get_max_record_seconds() -> float:
    return float(get_voice_setting("max_record_seconds", 30))


def get_vad_min_silence_ms() -> int:
    return int(get_voice_setting("vad_min_silence_ms", 500))


def get_vad_threshold() -> float:
    return float(get_voice_setting("vad_threshold", 0.5))


def get_stt_model_size() -> str:
    return str(get_voice_setting("stt_model_size", "base"))


def set_stt_model_size_override(size: str | None) -> None:
    if size:
        _VOICE["stt_model_size"] = size


def get_stt_language() -> str:
    return str(get_voice_setting("stt_language", "en"))


def get_stt_compute_type() -> str:
    return str(get_voice_setting("stt_compute_type", "int8"))


def get_stt_beam_size() -> int:
    return int(get_voice_setting("stt_beam_size", 5))


def get_stt_initial_prompt() -> str:
    return str(get_voice_setting("stt_initial_prompt", ""))


def get_stt_condition_on_previous_text() -> bool:
    return bool(get_voice_setting("stt_condition_on_previous_text", False))


def get_stt_vad_filter() -> bool:
    return bool(get_voice_setting("stt_vad_filter", False))


def get_speech_pad_ms() -> int:
    return int(get_voice_setting("speech_pad_ms", 150))


def get_audio_target_rms() -> float:
    return float(get_voice_setting("audio_target_rms", 0.08))


def get_piper_voice_id() -> str:
    return str(get_voice_setting("piper_voice_id", "en_US-lessac-medium"))


def get_piper_download_voice() -> str:
    return str(get_voice_setting("piper_download_voice", "en_US-lessac-medium"))


def get_ollama_num_predict() -> int:
    return int(get_voice_setting("ollama_num_predict", 400))


def get_default_ollama_model() -> str:
    data = _load_json(OLLAMA_MODELS_PATH)
    return str(data.get("default", "qwen2.5:3b"))


def piper_model_paths() -> tuple[Path, Path]:
    voice_id = get_piper_voice_id()
    model_path = MODELS_DIR / f"{voice_id}.onnx"
    config_path = MODELS_DIR / f"{voice_id}.onnx.json"
    return model_path, config_path
