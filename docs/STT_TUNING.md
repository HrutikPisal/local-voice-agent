# STT accuracy tuning

Pipeline: mic → Silero VAD → trim + RMS normalize (`voice/audio_prep.py`) → cached faster-whisper (`voice/stt.py`).

## Config keys (`voice/config.json`)

| Key | Default | Notes |
| --- | --- | --- |
| `stt_model_size` | `small` | Cached in-process; use `base` if RAM is tight |
| `stt_beam_size` | `5` | Higher = slower, often clearer |
| `stt_initial_prompt` | name/CLI hint | Helps names and short phrases |
| `stt_condition_on_previous_text` | `false` | Avoids run-on hallucinations per utterance |
| `stt_vad_filter` | `false` | Silero already endpoints; enable for noisy LiveKit clips |
| `speech_pad_ms` | `150` | Padding after Silero trim |
| `audio_target_rms` | `0.08` | Peak-safe normalize before STT |
| `vad_min_silence_ms` | `500` | Try 700–900 if cuts off early; 400–500 if trailing noise |
| `vad_threshold` | `0.5` | Lower = more sensitive to quiet speech |

Override model from CLI: `python -m voice --stt-model base`. Restart the REPL after config edits.

**RAM:** `small` + cached Whisper uses roughly 1–2 GB while `python -m voice` runs.

## Measure changes

```bash
python -m voice.tools.log_stats
python -m voice.benchmark_stt
```

Audit log fields on `voice_stt_result`: `stt_beam_size`, `audio_duration_sec`, `language_probability`, `avg_no_speech_prob`. First load logs `voice_stt_model_ready` with `stt_model_load_ms`.
