"""Generate labeled 16 kHz mono WAV fixtures for STT benchmark (Windows SAPI)."""

from __future__ import annotations

import json
import struct
import subprocess
import sys
import wave
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLES_DIR = REPO_ROOT / "voice" / "tests" / "fixtures" / "samples"
MANIFEST = REPO_ROOT / "voice" / "tests" / "fixtures" / "stt_manifest.json"

FIXTURES = [
    ("hi.wav", "Hi"),
    ("name_hrutik.wav", "Hi, I'm Hrutik"),
    ("stop.wav", "stop"),
    ("git_commit.wav", "git commit -m fix the voice agent"),
    ("run_pytest.wav", "run pytest on the voice tests"),
]


def _sapi_windows(wav_path: Path, text: str) -> bool:
    ps = f"""
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$s.SetOutputToWaveFile('{wav_path.resolve()}')
$s.Speak('{text.replace("'", "''")}')
$s.Dispose()
"""
    try:
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps],
            check=True,
            capture_output=True,
            timeout=60,
        )
        return wav_path.exists()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _resample_to_16k_mono(in_path: Path, out_path: Path) -> None:
    import numpy as np

    with wave.open(str(in_path), "rb") as wf:
        nch = wf.getnchannels()
        rate = wf.getframerate()
        width = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())
    if width != 2:
        raise ValueError("Expected 16-bit WAV from SAPI")

    samples = np.frombuffer(frames, dtype=np.int16)
    if nch == 2:
        samples = samples[0::2]

    if rate != 16000:
        target_len = int(len(samples) * 16000 / rate)
        x_old = np.linspace(0, 1, len(samples), endpoint=False)
        x_new = np.linspace(0, 1, target_len, endpoint=False)
        floats = samples.astype(np.float32)
        resampled = np.interp(x_new, x_old, floats)
        samples = np.clip(resampled, -32768, 32767).astype(np.int16)

    with wave.open(str(out_path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(16000)
        out.writeframes(samples.tobytes())


def main() -> int:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    if sys.platform != "win32":
        print("build_stt_fixtures: Windows SAPI only. Place 16 kHz mono WAVs in", SAMPLES_DIR)
        return 1

    manifest = []
    tmp_dir = SAMPLES_DIR / "_tmp"
    tmp_dir.mkdir(exist_ok=True)

    for filename, text in FIXTURES:
        raw = tmp_dir / filename
        final = SAMPLES_DIR / filename
        if final.exists():
            manifest.append({"path": f"samples/{filename}", "expected_text": text})
            continue
        if not _sapi_windows(raw, text):
            print(f"Failed to synthesize: {text}")
            return 1
        _resample_to_16k_mono(raw, final)
        manifest.append({"path": f"samples/{filename}", "expected_text": text})
        print("Wrote", final)

    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Updated manifest:", MANIFEST)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())