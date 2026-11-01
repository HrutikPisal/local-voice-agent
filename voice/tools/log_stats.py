"""Summarize voice audit logs (JSONL in logs/voice_*.txt)."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from voice.config import LOGS_DIR


def _iter_events(paths: list[Path]):
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = int(round((pct / 100.0) * (len(ordered) - 1)))
    return ordered[max(0, min(idx, len(ordered) - 1))]


def summarize(paths: list[Path]) -> dict:
    confirms: list[dict] = []
    summaries: list[dict] = []
    stt_ms: list[float] = []
    record_ms: list[float] = []
    total_ms: list[float] = []
    pairs: Counter[tuple[str, str]] = Counter()

    for row in _iter_events(paths):
        event = row.get("event")
        if event == "voice_confirm" and row.get("outcome") == "confirmed":
            confirms.append(row)
            raw = (row.get("transcript_raw") or "").strip()
            confirmed = (row.get("confirmed_text") or "").strip()
            if raw and confirmed and raw != confirmed:
                pairs[(raw, confirmed)] += 1
        if event == "voice_turn_summary":
            summaries.append(row)
            metrics = row.get("metrics") or {}
            for key, bucket in (
                ("stt_ms", stt_ms),
                ("record_ms", record_ms),
                ("total_ms", total_ms),
            ):
                val = metrics.get(key)
                if isinstance(val, (int, float)):
                    bucket.append(float(val))

    edited = sum(1 for c in confirms if c.get("was_edited"))
    confirm_n = len(confirms)
    edited_pct = (100.0 * edited / confirm_n) if confirm_n else 0.0

    return {
        "files": [str(p) for p in paths],
        "confirmed_turns": confirm_n,
        "was_edited_pct": round(edited_pct, 1),
        "median_stt_ms": round(statistics.median(stt_ms), 2) if stt_ms else None,
        "median_record_ms": round(statistics.median(record_ms), 2) if record_ms else None,
        "p95_total_ms": round(_percentile(total_ms, 95) or 0, 2) if total_ms else None,
        "top_mishearings": [
            {"stt": a, "confirmed": b, "count": n}
            for (a, b), n in pairs.most_common(10)
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stats from logs/voice_*.txt")
    parser.add_argument(
        "logs",
        nargs="*",
        help="Log files (default: all logs/voice_*.txt)",
    )
    args = parser.parse_args(argv)

    if args.logs:
        paths = [Path(p) for p in args.logs]
    else:
        paths = sorted(LOGS_DIR.glob("voice_*.txt"))

    if not paths:
        print("No voice log files found.")
        return 1

    report = summarize(paths)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
