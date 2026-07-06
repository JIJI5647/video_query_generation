# Caption → Query integration test

Three independent, cacheable scripts check that a **new caption model can plug
into the existing pipeline**. Each stage reads the prior stage's output dir, so
any stage can be re-run on its own (e.g. iterate on query-generation without
re-running GPU caption inference):

```
run_caption_generation_test.py    caption model → normalized captions (+ segments.jsonl)
run_query_generation_test.py      captions → Gemini emotion-event → Gemini query-generation
run_evaluation_test.py            queries → Gemini verify/rewrite → final_queries.json
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
| `avocado` | AVoCaDO AV caption (Qwen2.5-Omni-7B fine-tune) | `--video` | implemented (shared `_run_qwen2_5_omni_av` runner, in-process) |
| `timechat` | TimeChat-Captioner-GRPO-7B AV caption (timestamps; Qwen2.5-Omni-7B fine-tune) | `--video` | implemented (shared `_run_qwen2_5_omni_av` runner, in-process) |
| `af3_vl` | Audio Flamingo 3 (audio) + `Qwen3-VL-8B` (video) | `--video --audio` | implemented — AF3's audio half needs a newer `transformers` than the shared env, so it runs as a **subprocess** in `conda_envs/af3_env` via `standalone_runners/af3_infer.py`; video half in-process |
| `secap_qwen` | SECap (speech/audio-emotion, used directly) + `Qwen3-VL-8B` (video) | `--video --audio` | implemented — SECap needs a 2023 legacy torch/transformers pin, so it runs as a **subprocess** in `conda_envs/secap_env` via `third_party/SECap/scripts/standalone_inference.py`; video half in-process; never calls Qwen3-Omni-Captioner |

All six runners are wired end-to-end; see `docs/progress_log.md` for which have actually
been exercised on real GPU inference vs. still implemented-but-unverified.

## Example commands

`--max-segments 1` (default) treats `--video`/`--audio` as a pre-cut clip — no real
segmentation. `--max-segments N > 1` cuts real ffmpeg segments (`--segment-seconds`,
default 5s) and, for audio+video models, a matching per-segment audio slice
extracted from each cut clip — this is what actually exercises the real pipeline's
segmentation, not just "does the model's output parse".

```bash
# qwen3_omni (AV, single model) — 3 real 5s segments
python run_caption_generation_test.py --caption-model qwen3_omni \
  --video short.mp4 --output output/caption_query_tests/qwen3_omni --max-segments 3
python run_query_generation_test.py \
  --captions-dir output/caption_query_tests/qwen3_omni \
  --output output/caption_query_tests/qwen3_omni

# qwen_audio_vl (audio + video, merged)
python run_caption_generation_test.py --caption-model qwen_audio_vl \
  --video short.mp4 --audio short.wav \
  --output output/caption_query_tests/qwen_audio_vl --max-segments 3
python run_query_generation_test.py \
  --captions-dir output/caption_query_tests/qwen_audio_vl \
  --output output/caption_query_tests/qwen_audio_vl

# af3_vl (Audio Flamingo 3 audio + Qwen3-VL video) — NON-COMMERCIAL research only
python run_caption_generation_test.py --caption-model af3_vl \
  --video short.mp4 --audio short.wav \
  --output output/caption_query_tests/af3_vl --max-segments 3

# secap_qwen (SECap audio + Qwen3-VL video)
python run_caption_generation_test.py --caption-model secap_qwen \
  --video short.mp4 --audio short.wav \
  --output output/caption_query_tests/secap_qwen --max-segments 3

# stage 3, on any of the above once stage 2 produced queries:
python run_evaluation_test.py \
  --queries-dir output/caption_query_tests/qwen3_omni \
  --output output/caption_query_tests/qwen3_omni
```

## Output files

Written incrementally into the same `--output` dir as stages run:

| File | Written by | Meaning |
|---|---|---|
| `raw_caption_output.json` | stage 1 | raw caption-model output(s); for audio+video models, both `audio_text` and `video_text` |
| `normalized_captions.jsonl` | stage 1 | captions coerced into the pipeline `OmniCaption` schema (one per segment) |
| `segments.jsonl` | stage 1 (written), 2 (passthrough) | `Segment` list incl. `clip_path`, so later stages never need to re-cut |
| `run_metadata.json` | stage 1 | model, paths, counts, timings |
| `emotion_events.json` | stage 2 | Gemini emotion-event stage output (`EmotionEventOutput`) |
| `generated_queries.json` | stage 2 | Gemini query-generation output (`GenerationOutput`) |
| `generation_metadata.json` | stage 2 | counts, timings, warnings (e.g. **zero queries**) |
| `final_queries.json` | stage 3 | per-query trace after verify/rewrite |
| `verification_summary.json` | stage 3 | pass/revise/fail counts |
| `evaluation_metadata.json` | stage 3 | model, counts |

If Gemini returns **zero queries**, that is recorded explicitly in
`generation_metadata.json → warnings` (never hidden).

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
