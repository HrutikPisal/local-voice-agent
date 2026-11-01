"""Download Piper voice assets on first use."""

from __future__ import annotations

import subprocess
import sys
import urllib.request
from pathlib import Path

from voice.config import MODELS_DIR, get_piper_download_voice, get_piper_voice_id, piper_model_paths

HF_BASE = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium"
)


def _download_url(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[VOICE] Downloading {dest.name}...")
    urllib.request.urlretrieve(url, dest)


def ensure_models() -> None:
    """Ensure Piper ONNX + config exist under voice/models/voices/."""
    model_path, config_path = piper_model_paths()
    if model_path.exists() and config_path.exists():
        return

    voice_id = get_piper_voice_id()
    try:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [sys.executable, "-m", "piper.download_voices", get_piper_download_voice()],
            check=True,
            cwd=str(MODELS_DIR),
        )
        if model_path.exists() and config_path.exists():
            return
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    _download_url(f"{HF_BASE}/{voice_id}.onnx", model_path)
    _download_url(f"{HF_BASE}/{voice_id}.onnx.json", config_path)

    if not model_path.exists() or not config_path.exists():
        raise RuntimeError(
            f"Could not download Piper voice '{voice_id}'. "
            "Check your network or run: python -m piper.download_voices "
            f"{get_piper_download_voice()}"
        )
