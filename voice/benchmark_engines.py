"""Compare STT engines / model sizes on fixture WAVs (CPU)."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from voice.audio_prep import prepare_for_stt
from voice.benchmark_stt import DEFAULT_MANIFEST, _load_wav, _wer, run_benchmark
from voice.config import LOGS_DIR, set_stt_model_size_override
from voice.stt import transcribe_with_metadata
from voice.stt_engine import clear_engine_cache, create_stt_engine


def bench_models(models: list[str], manifest_path: Path) -> dict:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = data if isinstance(data, list) else data.get("samples", [])
    base = manifest_path.parent
    rows = []

    for model in models:
        clear_engine_cache()
        set_stt_model_size_override(model)
        engine = create_stt_engine(model_size=model)
        load_info = engine.load()
        load_ms = load_info.stt_model_load_ms if load_info else 0.0

        wer_sum = 0.0
        lat_sum = 0.0
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
                continue
            audio = _load_wav(wav_path)
            prepared, _, _ = prepare_for_stt(audio)
            t0 = time.perf_counter()
            dec = engine.transcribe(prepared)
            lat_sum += (time.perf_counter() - t0) * 1000
            wer_sum += _wer(expected, dec.text)
            ran += 1

        rows.append(
            {
                "model": model,
                "load_ms": round(load_ms, 1),
                "mean_decode_ms": round(lat_sum / ran, 1) if ran else None,
                "mean_wer": round(wer_sum / ran, 4) if ran else None,
                "samples": ran,
            }
        )

    return {"manifest": str(manifest_path), "models": rows}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark Whisper model sizes on fixtures")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["base.en", "small.en", "distil-small.en"],
    )
    parser.add_argument("--baseline-json", action="store_true", help="Also run default transcribe_with_metadata report")
    args = parser.parse_args(argv)

    if not args.manifest.exists():
        print(f"Manifest not found: {args.manifest}")
        return 1

    report = bench_models(args.models, args.manifest)
    if args.baseline_json:
        report["default_pipeline"] = run_benchmark(args.manifest)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    out = LOGS_DIR / f"stt_engine_benchmark_{stamp}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("Saved:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())