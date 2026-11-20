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

## Recent tuning changes

### Optimised `stt_initial_prompt` for CLI/coding vocabulary

The old prompt was a natural-language description ("The user speaks short
commands…"). Whisper treats `initial_prompt` as *preceding transcript text*, so
it works better as a dense token list than as an instruction sentence.

The new prompt (`voice/config.json`):
```
CLI coding assistant. Git: commit, push, pull, diff, merge, rebase, branch, stash, checkout. Python: def, class, import, return, async, await, pip, venv. ls, cd, mkdir, rm, grep. PR, API, SSH, env, stdout, stderr.
```

This seeds the BPE context with tokens that share byte-pairs with common
developer terms, reducing mis-transcription of jargon (e.g. "rebase" → "re
base", "venv" → "van", "stderr" → "step there").  The prompt is kept under
~200 characters because longer initial prompts can suppress genuine speech via
the no-speech gate.

### `no_speech_prob` filter in `transcribe_with_metadata()`

After `avg_no_speech` is computed, a guard is applied:

```python
if avg_no_speech is not None and avg_no_speech > 0.7 and len(text.split()) < 3:
    return TranscriptionResult(text="", avg_no_speech_prob=avg_no_speech, ...), load_info
```

**Rationale:** When Whisper's internal VAD assigns > 70 % probability that a
chunk is silence/noise *and* has produced fewer than 3 words, the result is
almost certainly a hallucination (e.g. "Thank you." on a silent frame).
Returning `text=""` early lets callers skip the confirm prompt.

**How callers detect this case:** `result.text == ""` **and**
`result.avg_no_speech_prob is not None` (the value will be > 0.7).

Threshold choice: 0.7 is conservative — real speech with a noisy mic rarely
exceeds 0.5. Lower to 0.6 if false-positives occur in very noisy environments.

### `word_timestamps` kwarg on `transcribe_with_metadata()`

```python
result, load_info = transcribe_with_metadata(audio, word_timestamps=True)
# Each segment in the underlying faster-whisper response now has a .words list
# with per-word start/end times and probabilities.
```

Defaults to `False` (unchanged behaviour).  Enable for future features such as
karaoke-style highlighting or per-word confidence gating.  No config.json key
is needed; callers opt in at the call site.

## Latency defaults (CPU)

| Key | Default | Notes |
| --- | --- | --- |
| `stt_model_size` | `base.en` | English-only; use `distil-small.en` or `small.en` for accuracy |
| `stt_beam_size` | `2` | Final decode; partials use `stt_partial_beam_size` |
| `stt_cpu_threads` | `4` | CTranslate2 CPU threads |
| `vad_min_silence_ms` | `400` | Adaptive endpoint may extend after fillers |
| `record_preroll_ms` | `300` | Pre-roll so word onsets are not clipped |
| `stt_partial_enabled` | `true` | Live partial transcript while speaking |

CLI preload (default on): background model load + warm-up decode. Disable with `python -m voice --no-preload`.

## Engines (`stt_engine`)

| Value | When |
| --- | --- |
| `faster_whisper` | Default; install `requirements-stt.txt` |
| `sherpa_onnx` | Set `stt_sherpa_model_dir` to a transducer model tree |
| `moonshine_onnx` | `pip install moonshine-onnx`; set `stt_model_size` to `tiny` or `base` |

Compare models: `python -m voice.benchmark_engines`

## Hotwords

Edit [config/hotwords.json](../config/hotwords.json) for post-STT replacements and extra prompt terms (git, pytest, etc.).

## Confirm UX

| Config | Default |
| --- | --- |
| `confirm_auto_accept_enabled` | `false` — set `true` for high-confidence skip |
| `confirm_auto_accept_min_logprob` | `-0.35` |
| `confirm_auto_accept_max_no_speech` | `0.35` |

Transcript prompt commands: `.cancel`, `.rerecord`, `.append`

## Baseline benchmark (fixtures)

Generate WAVs (Windows): `python -m voice.tools.build_stt_fixtures`

Run WER: `python -m voice.benchmark_stt`

SAPI synthetic speech is a rough baseline; replace fixtures with your own 16 kHz mono recordings for production tuning.
