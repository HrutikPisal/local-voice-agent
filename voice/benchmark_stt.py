"""Run faster-whisper on labeled WAV fixtures; report WER and exact match."""

from __future__ import annotations

import argparse
import json
import wave
from datetime import datetime
from pathlib import Path

import numpy as np

from voice.audio_prep import prepare_for_stt
from voice.config import LOGS_DIR, REPO_ROOT
from voice.stt import transcribe_with_metadata

DEFAULT_MANIFEST = REPO_ROOT / "voice" / "tests" / "fixtures" / "stt_manifest.json"


def _load_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wf:
        if wf.getnchannels() != 1:
            raise ValueError(f"Expected mono WAV: {path}")
        if wf.getsampwidth() != 2:
            raise ValueError(f"Expected 16-bit PCM: {path}")
        frames = wf.readframes(wf.getnframes())
    pcm = np.frombuffer(frames, dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().strip().split())


def _wer(reference: str, hypothesis: str) -> float:
    ref = _normalize_text(reference).split()
    hyp = _normalize_text(hypothesis).split()
    if not ref:
        return 0.0 if not hyp else 1.0
    # Levenshtein at word level
    rows = len(ref) + 1
    cols = len(hyp) + 1
    dist = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        dist[i][0] = i
    for j in range(cols):
        dist[0][j] = j
    for i in range(1, rows):
        for j in range(1, cols):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dist[i][j] = min(
                dist[i - 1][j] + 1,
                dist[i][j - 1] + 1,
                dist[i - 1][j - 1] + cost,
            )
    return dist[rows - 1][cols - 1] / len(ref)


def run_benchmark(manifest_path: Path) -> dict:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = data if isinstance(data, list) else data.get("samples", [])
    base = manifest_path.parent

    results = []
    exact = 0
    wer_sum = 0.0
    ran = 0

    for item in entries:
        rel = item.get("path") or item.get("wav")
        expected = item.get("expected_text") or item.get("text") or ""
        if not rel:
            continue
        wav_path = Path(rel)
        if not wav_path.is_absolute():
            wav_path = base / wav_path
        if not wav_path.exists():
            results.append({"path": str(wav_path), "skipped": "missing file"})
            continue

        audio = _load_wav(wav_path)
        prepared, trimmed, duration = prepare_for_stt(audio)
        stt_result, _load_info = transcribe_with_metadata(prepared)
        hyp = stt_result.text
        w = _wer(expected, hyp)
        match = _normalize_text(expected) == _normalize_text(hyp)
        if match:
            exact += 1
        wer_sum += w
        ran += 1
        results.append(
            {
                "path": str(wav_path),
                "expected": expected,
                "hypothesis": hyp,
                "wer": round(w, 4),
                "exact_match": match,
                "trimmed_samples": trimmed,
                "audio_duration_sec": round(duration, 3),
                "language_probability": stt_result.language_probability,
            }
        )

    report = {
        "manifest": str(manifest_path),
        "ran": ran,
        "exact_match_rate": round(exact / ran, 4) if ran else None,
        "mean_wer": round(wer_sum / ran, 4) if ran else None,
        "samples": results,
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="STT benchmark from WAV manifest")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="JSON manifest with path + expected_text",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write JSON report (default: logs/stt_benchmark_<timestamp>.json)",
    )
    args = parser.parse_args(argv)

    if not args.manifest.exists():
        print(f"Manifest not found: {args.manifest}")
        return 1

    report = run_benchmark(args.manifest)
    out = args.out
    if out is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        out = LOGS_DIR / f"stt_benchmark_{stamp}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("ran", "exact_match_rate", "mean_wer")}, indent=2))
    print(f"Full report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
