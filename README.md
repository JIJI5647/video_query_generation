# Caption-based Emotion Query Generation Pipeline (v4)

Generates emotion-related **temporal grounding queries** from raw videos by first
building per-segment **emotion captions**, then writing queries from the caption text
(plus the spoken-dialogue transcript). Each query is grounded to a **time range**
`[start, end]` of the video, so grounding is tied to specific moments rather than the
whole video.

This is a self-contained sibling of `video_query_answering_demo` (the v1 demo); it copies
the segmentation/clip-extraction utilities and the verify/rewrite stage rather than
importing them.

## Flow

```
raw .mp4
  └─ 1. ffprobe duration + cut fixed 5s segments (s001, s002, ...) with ffmpeg,
        into a PERSISTENT cache: data/processed_segments/<video_id>/<grid_key>/
        (reused across runs; never auto-deleted)                          [B4]
  └─ 2. group segments into batches (default 8 = 40s)   [gemini backend only]
  └─ 3. captions, via one of two backends:
        • qwen3_omni (default): ONE segment per prompt → structured caption,
                                cached per-segment for resume (see below)
        • gemini              : per batch, upload N clips → ONE multimodal call
  └─ 4. WhisperX transcribes the whole video → sentence-level text + timestamps [B3]
  └─ 5. caption-only LLM call sees ALL of the video's captions (with time ranges)
        + the dialogue transcript → few, high-precision queries, each grounded
        on a time_range. Caption `emotion` is treated as a candidate, not a
        must-copy label.                                                   [B1,B2]
  └─ 6. for each query, take ONLY its grounded segment clip(s) → verify ⇄ rewrite
        (Gemini upload, or watched locally on Qwen3-Omni).
        The verifier sees ONLY query_id + query_text (no caption metadata).  [A1]
  └─ export intermediates + final queries; segment clips stay cached, uploads cleaned
```

Step 3 sends the actual clips (with audio); step 5 sends **only** text (captions +
transcript, no video). There is no caption-filtering step — every caption is fed to
generation, which selects which moments are worth a query. `segment_id` remains an internal handle that maps a query's
`time_range` back to the segment clip(s) used for verification — it is never shown to the
models or to human annotators. The eight emotion labels are fixed: `angry, excited, fear,
sad, surprised, frustrated, happy, disappointed`.

The v4 work targets verification (A1–A4), generation grounding (B1), caption-emotion
handling (B2), audio transcription (B3), and the segment cache (B4). A later addition adds
a second, pluggable **caption backend** (`qwen3_omni`, see below) alongside the original
Gemini batch captioner. Prompt versions: `verification_prompt_v7`,
`generation_prompt_caption_v6`, `omni_caption_prompt_v1`.

## Setup

```bash
pip install -r requirements.txt   # google-genai, pydantic>=2
pip install whisperx              # transcription (B3); pulls torch CPU + faster-whisper
export GEMINI_API_KEY="..."       # required
# ffmpeg + ffprobe must be on PATH (clip cutting + duration probing + WhisperX audio)
```

WhisperX is heavy (~2–3 GB incl. torch) and downloads its ASR + alignment models on first
run. Run with `--no-transcript` to skip it entirely.

The `qwen3_omni` caption backend additionally needs `transformers`,
`qwen-omni-utils`, `torch` and `accelerate` (plus `vllm` for the default vLLM
engine, or `flash-attn` for faster attention) — install these **only on the
inference server**, kept out of the main requirements so the default pipeline and
tests stay light:

```bash
pip install -r requirements.txt -r requirements-qwen.txt   # GPU server only
```

Install the `torch` build that matches your GPU driver's CUDA. They are imported
lazily (only on the first inference call), so the rest of the pipeline and all
tests run without them. The `transformers` engine does not need `vllm` at all.

## Run

```bash
python run_pipeline.py \
  --video-dir data/pilot_study \
  --video-ids "emostim_01_TheShining_clip_2,meld_01_dia337" \
  --output output/pilot_study_v4
```

Selection: `--video-ids` pins an exact set (file stems, comma/space-separated, in order)
and overrides the default `--num-videos`/`--seed` random sampling.

Key flags (all optional except `--video-dir`):
`--num-videos --seed --video-ids --batch-size --segment-seconds --stride`
`--max-rewrites --max-accepted`
`--caption-model --generation-model --verification-model --rewrite-model`
`--segments-dir --force-reextract --no-transcript --whisper-model`
`--caption-backend --caption-batch-size --qwen-model-path --qwen-engine --qwen-attn-impl`
`--qwen-video-reader-backend --verify-rewrite-backend --captions-cache-dir`
`--resume / --no-resume --overwrite-captions`

Defaults: 5s segments / 5s stride (non-overlapping), max_accepted 8, max_rewrites 3,
segments cached under `data/processed_segments`, WhisperX model `small`.

Backend defaults: **caption + verify/rewrite on Qwen3-Omni**
(`Qwen/Qwen3-Omni-30B-A3B-Instruct`, vLLM engine, caption batch 1), **generation on
Gemini** (`gemini-2.5-flash-lite`). The `--caption-model` / `--verification-model`
(`gemini-3.1-flash-lite`) / `--rewrite-model` / `--batch-size 8` flags only take
effect for whichever of those stages you switch back to Gemini.

### Caption backends

**By default, caption + verification + rewrite run on Qwen3-Omni; only generation
(query writing) runs on Gemini.** `GEMINI_API_KEY` is still required for generation.
Switch any stage back to Gemini with `--caption-backend gemini` /
`--verify-rewrite-backend gemini` (the Gemini caption path uses the Files API batch
above; Gemini verify/rewrite uploads clips).

```bash
# default: caption + verify/rewrite on Qwen3-Omni (vLLM engine), generation on Gemini
python run_pipeline.py \
  --video-dir data/pilot_study --num-videos 5 \
  --output output/pilot_study_v4 \
  --resume

# transformers engine (fallback when vLLM won't load Qwen3-Omni as multimodal)
python run_pipeline.py \
  --video-dir data/pilot_study --num-videos 5 \
  --output output/pilot_study_v4 \
  --qwen-engine transformers \
  --resume

# all stages on Gemini (the old behaviour)
python run_pipeline.py \
  --video-dir data/pilot_study --num-videos 5 \
  --output output/pilot_study_v4 \
  --caption-backend gemini --verify-rewrite-backend gemini \
  --resume
```

Qwen3-Omni specifics:

- **Model:** `Qwen/Qwen3-Omni-30B-A3B-Instruct` by default (override with
  `--qwen-model-path`), with audio-in-video enabled.
- **Engine (`--qwen-engine`):** `vllm` (default, fast) or `transformers` (slower
  pure-HuggingFace fallback). Use `transformers` when the installed vLLM build
  won't load Qwen3-Omni as a multimodal model on the available CUDA/driver (e.g.
  vLLM errors with *"`limit_mm_per_prompt` is only supported for multimodal
  models"*). The transformers engine loads with `device_map="auto"`, disables the
  audio "talker" for text-only output, and only needs a working torch — no vLLM.
  Pass `--qwen-attn-impl flash_attention_2` if flash-attn is installed.
- **Lazy load:** the model is loaded only on the first inference call — importing
  the module or constructing the backend touches no GPU and downloads no weights.
  Run the pipeline on a GPU server; it cannot run on a laptop.
- **Video reader (`--qwen-video-reader-backend`):** forces the `qwen_omni_utils`
  video reader (sets `FORCE_QWENVL_VIDEO_READER`). Default `torchvision` avoids
  `torchcodec`, which often fails to load on mismatched CUDA/ffmpeg. Equivalent to
  exporting `FORCE_QWENVL_VIDEO_READER=torchvision` yourself.
- **One segment per prompt:** each prompt always contains exactly one segment, so
  `segment_id` / `time_range` / caption can never be mis-paired. `--caption-batch-size`
  (default `1`, sequential) controls how many of these *independent* single-segment
  prompts are submitted in one batched model call for throughput — they are decoded
  per-index and never mixed within a prompt. Larger batches are faster but use more
  VRAM; if a batched call errors it degrades gracefully to per-segment. Resume is
  applied first, so only cache-miss segments are batched.
- **Structured output:** each caption is a nested JSON with `visual_objective`
  (objective facts only), `visual_expression` (observable facial/body/gaze cues),
  `audio_description` (non-transcript audio), `emotion_description` (a *candidate*
  reading), plus `confidence` / `evidence_strength`. It is adapted to the existing
  flat `EmotionCaption` for the rest of the pipeline, so generation/export are
  unchanged. The free-text `emotion_description` maps to one of the eight labels by
  keyword, falling back to `neutral`. There is no caption-filtering step, so every
  caption (including neutral ones) is fed to generation, which selects moments.

**Verify/rewrite on Qwen3-Omni (default):** the verification and rewrite stages
(both watch a query's grounded segment clip(s)) run on the **same** loaded
Qwen3-Omni model — clips are read from their local paths, with no Files API upload.
The model loads once and is shared with the caption backend. Generation always
stays on Gemini, so `GEMINI_API_KEY` is still required. Set
`--verify-rewrite-backend gemini` to put these two stages back on Gemini.

**Resume / cache:** each structured caption is written atomically to
`<output>/captions/<video_id>/<segment_id>.json`. On rerun, a segment with a valid
cached caption (parseable + all required fields) is **skipped without calling the
model** (`--resume`, on by default; disable with `--no-resume`).
`--overwrite-captions` forces regeneration. Invalid/missing cache is regenerated;
a parse failure saves the raw model text to
`<output>/captions_raw/<video_id>/<segment_id>.txt` for debugging.
Override the cache root with `--captions-cache-dir`.

### Segment cache (B4)

Segment clips are cut once into `data/processed_segments/<video_id>/<grid_key>/`, where
`grid_key` (e.g. `win5.00_str5.00`) encodes the windowing params so a different
segment-length/stride writes a fresh subdir and never reuses mismatched clips. Subsequent
runs reuse existing clips (no ffmpeg) unless `--force-reextract` is given. This keeps
verification deterministic, speeds up reruns, and lets you open a clip to debug a
caption/verification mismatch.

## Outputs (under `--output`)

| File | Contents |
|------|----------|
| `segments.jsonl` | every segment with its time span |
| `raw_captions.jsonl` | captions straight from the model (all fed to generation) |
| `initial_queries.jsonl` | queries generated from captions |
| `verification_rounds.jsonl` | every verifier round |
| `rewritten_queries.jsonl` | every rewrite |
| `final_queries.jsonl` | per-query trace + final status; carries `time_range` (external handle) and `segment_ids` (internal provenance) |
| `human_review_sheet.csv` | accepted queries for human review; `time_range` column (e.g. `65-70s`), no segment ids |
| `pipeline_stats.json` | aggregate stats |
| `prompts_used/` | the prompt templates + version manifest |
| `captions/<video_id>/<segment_id>.json` | structured Qwen3-Omni captions (resume cache; `qwen3_omni` backend only) |
| `captions_raw/<video_id>/<segment_id>.txt` | raw model text saved on a caption parse failure (`qwen3_omni` only) |

`rerun_generation.py` re-runs only generation+verification from a prior run's captions
(reusing the segment cache and re-transcribing); it takes the same B1/B3/B4 flags.

## Testing without API calls

The pure modules have no SDK imports and can be exercised directly:
`segmentation.plan_segments` / `grid_key`,
`generation._resolve_time_ranges` (validates `time_range` → covering `segment_ids`), and
`workflow.run_query_pipeline` (inject a fake `BaseLLMClient`). Only `llm_client`,
`captioning.GeminiUploader`, and `transcription` (WhisperX) touch external models.

The Qwen3-Omni backend is fully testable without the model — prompt construction,
JSON extraction, field validation, the cache/resume decision, atomic write and the
`OmniCaption → EmotionCaption` adapter are pure Python (heavy deps imported lazily):

```bash
python -m pytest tests/test_omni_captioning.py -q   # 22 tests, no GPU / no weights
```
