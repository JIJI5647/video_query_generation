# Caption → Query integration test

`run_caption_query_test.py` checks that a **new caption model can plug into the
existing pipeline**. It runs a caption model on a short clip, normalizes its
output into the pipeline's `OmniCaption` schema, and feeds those captions through
the **real** Gemini emotion-event + query-generation stages to produce queries.

```
caption model → normalized captions → Gemini emotion-event → Gemini query-generation → generated_queries.json
```

## What it checks / does not check

- **Checks:** a caption model's output can flow through the existing Gemini
  downstream and yield validated queries — an *integration/plumbing* check.
- **Does NOT check:** caption quality, query quality, grounding correctness, or
  any benchmark/human evaluation. It is not a leaderboard.

### Three related things (don't confuse them)

| | What it runs | What it proves |
|---|---|---|
| **Caption-only test** | just the caption model | the model produces a caption |
| **Caption→query integration** (this) | caption model + **real Gemini** downstream | captions flow through Gemini and produce queries |
| **Full pipeline benchmark** (`run_pipeline.py`) | full pipeline over many videos + scoring | end-to-end quality across a dataset |

## Supported caption models

| `--caption-model` | Type | Inputs | Status |
|---|---|---|---|
| `qwen3_omni` | Omni AV baseline (`Qwen/Qwen3-Omni-30B-A3B-Instruct`) | `--video` | implemented (reuses the pipeline captioner) |
| `qwen_audio_vl` | `Qwen3-Omni-Captioner` (audio, no text prompt) + `Qwen3-VL-8B` (video) | `--video --audio` | implemented (both halves) |
| `af3_vl` | Audio Flamingo 3 (audio) + `Qwen3-VL-8B` (video) | `--video --audio` | video half implemented; **AF3 audio half raises a clear `NotImplementedError` with setup instructions** |
| `secap_qwen` | SECap (speech/audio-emotion, used directly) + `Qwen3-VL-8B` (video) | `--video --audio` | video half implemented; **SECap audio half raises a clear `NotImplementedError`**; never calls Qwen3-Omni-Captioner |
| `avocado` | AVoCaDO AV caption | `--video` | **`NotImplementedError`** (add the AVoCaDO repo runner) |
| `timechat` | TimeChat AV caption (timestamps) | `--video` | **`NotImplementedError`** (add the TimeChat repo runner) |

## Example commands

```bash
# qwen3_omni (AV, single model)
python run_caption_query_test.py --caption-model qwen3_omni \
  --video short.mp4 --output output/caption_query_tests/qwen3_omni \
  --max-segments 1 --generation-model gemini-2.5-flash-lite

# qwen_audio_vl (audio + video, merged)
python run_caption_query_test.py --caption-model qwen_audio_vl \
  --video short.mp4 --audio short.wav \
  --output output/caption_query_tests/qwen_audio_vl --max-segments 1

# af3_vl (Audio Flamingo 3 audio + Qwen3-VL video) — NON-COMMERCIAL research only
python run_caption_query_test.py --caption-model af3_vl \
  --video short.mp4 --audio short.wav \
  --output output/caption_query_tests/af3_vl --max-segments 1

# secap_qwen (SECap audio + Qwen3-VL video)
python run_caption_query_test.py --caption-model secap_qwen \
  --video short.mp4 --audio short.wav \
  --output output/caption_query_tests/secap_qwen --max-segments 1
```

`--with-verification` additionally runs the Gemini verify/rewrite loop on the
grounded clips and writes `final_queries.json`.

## Output files

All under `--output`:

| File | Meaning |
|---|---|
| `raw_caption_output.json` | raw caption-model output(s); for audio+video models, both `audio_text` and `video_text` |
| `normalized_captions.jsonl` | captions coerced into the pipeline `OmniCaption` schema (one per segment) |
| `emotion_events.json` | Gemini emotion-event stage output (`EmotionEventOutput`) |
| `generated_queries.json` | Gemini query-generation output (`GenerationOutput`) |
| `run_metadata.json` | model, paths, counts, timings, and any warnings (e.g. **zero queries**) |
| `final_queries.json` | only with `--with-verification`: per-query trace after verify/rewrite |

If Gemini returns **zero queries**, that is recorded explicitly in
`run_metadata.json → warnings` (never hidden).

## Required env vars

- `GEMINI_API_KEY` — always required (the downstream stages call Gemini).
- Integration test only: `RUN_CAPTION_QUERY_INTEGRATION=1`, plus
  `CAPTION_QUERY_MODEL`, `CAPTION_QUERY_VIDEO`, `CAPTION_QUERY_AUDIO`,
  `CAPTION_QUERY_OUTPUT`.

## Notes / boundaries

- **Audio Flamingo 3 is NON-COMMERCIAL research only.**
- **Qwen3-Omni-Captioner is audio-only and takes no text prompt** — the audio
  turn carries just the audio.
- **Qwen3-VL is video/image + text and must not receive audio.**
- **`secap_qwen` uses SECap directly for audio evidence — it never calls
  Qwen3-Omni-Captioner.**
- Heavy deps (`torch`, `transformers`, `qwen_omni_utils`, `qwen_vl_utils`,
  `decord`, `soundfile`, model repos) are imported lazily inside the runners;
  importing the script/module loads none of them and needs no GPU / no Gemini key.

## Local (offline) tests

```bash
python -m pytest tests/test_caption_query_utils.py -q          # no models, no Gemini
# keep the existing key tests green too:
python -m pytest tests/test_omni_captioning.py tests/test_generation.py \
                 tests/test_caption_query_utils.py -q
```
