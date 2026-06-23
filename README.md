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
  └─ 2. captions, --caption-batch-size segments/prompt (default 1), one of two backends:
        • qwen3_omni (default): N segments per prompt → JSON array of N structured
                                captions (mapped back to segment_ids), cached
                                per-segment for resume; --caption-parallel prompts
                                per generate() call
        • gemini              : upload N clips → ONE multimodal call
  └─ 3. WhisperX transcribes the whole video → sentence-level text + timestamps [B3]
  └─ 4. caption-only LLM call sees ALL of the video's captions (with time ranges)
        + the dialogue transcript → few, high-precision queries, each grounded
        on a time_range. Caption `emotion` is treated as a candidate, not a
        must-copy label.                                                   [B1,B2]
  └─ 5. for each query, take ONLY its grounded segment clip(s) → verify ⇄ rewrite
        (Gemini upload, or watched locally on Qwen3-Omni).
        The verifier sees ONLY query_id + query_text (no caption metadata).  [A1]
  └─ export intermediates + final queries; segment clips stay cached, uploads cleaned
```

Step 2 sends the actual clips (with audio); step 4 sends **only** text (captions +
transcript, no video). There is no caption-filtering step — every caption is fed to
generation, which selects which moments are worth a query. `segment_id` remains an internal handle that maps a query's
`time_range` back to the segment clip(s) used for verification — it is never shown to the
models or to human annotators. The eight emotion labels are fixed: `angry, excited, fear,
sad, surprised, frustrated, happy, disappointed`.

The v4 work targets verification (A1–A4), generation grounding (B1), caption-emotion
handling (B2), audio transcription (B3), and the segment cache (B4). A later addition adds
a second, pluggable **caption backend** (`qwen3_omni`, see below) alongside the original
Gemini batch captioner. Prompt versions: `verification_prompt_v11`,
`generation_prompt_caption_v9`, `omni_caption_prompt_v2`.

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

### Flags

All optional except `--video-dir`. Grouped by purpose; default in **bold**.

**Input / selection**

| Flag | Default | Meaning |
|------|---------|---------|
| `--video-dir` | *(required)* | Directory of source `.mp4` / `.avi` videos. |
| `--num-videos`, `-n` | **all** | How many videos to sample. Omit to process every video in `--video-dir`. |
| `--video-ids` | **None** | Comma/space-separated file stems to process exactly, in order. Overrides `--num-videos`/`--seed`. |
| `--seed` | **42** | Random seed for video sampling. |
| `--output` | **output/v2_run** | Output directory for all artefacts. |

**Segmentation / clips**

| Flag | Default | Meaning |
|------|---------|---------|
| `--segment-seconds` | **5.0** | Length of each fixed segment, in seconds. |
| `--stride` | **5.0** | Hop between segment starts (= `segment-seconds` → non-overlapping). |
| `--min-segment-seconds` | **1.0** | Drop a trailing segment shorter than this. |
| `--segments-dir` | **data/processed_segments** | Persistent segment-clip cache root (reused across runs, never auto-deleted). |
| `--force-reextract` | **off** | Re-cut segment clips even if cached copies exist. |
| `--temp-dir` | **temp_clips** | Scratch dir for clips. |
| `--keep-temp-clips` | **off** | Keep temp clips instead of cleaning them up. |

**Caption backend**

| Flag | Default | Meaning |
|------|---------|---------|
| `--caption-backend` | **qwen3_omni** | `qwen3_omni` (local Qwen3-Omni) or `gemini` (Files API). |
| `--caption-batch-size` | **1** | Segments per caption prompt (both backends). The model sees N segment clips in one prompt and returns N captions mapped back to their segment_ids. Larger = fewer prompts but a bigger single prompt. |
| `--caption-parallel` | **1** | qwen3_omni only: how many caption prompts run in ONE batched `generate` call (throughput). Orthogonal to `--caption-batch-size`; larger uses more VRAM. |

**Qwen3-Omni model / engine** (used when any stage is `qwen3_omni`)

| Flag | Default | Meaning |
|------|---------|---------|
| `--qwen-model-path` | **Qwen/Qwen3-Omni-30B-A3B-Instruct** | HF model id / local path. |
| `--qwen-engine` | **vllm** | `vllm` (fast; needs CUDA-matched vLLM with Qwen3-Omni multimodal) or `transformers` (slower HF fallback; only needs a working torch). |
| `--qwen-attn-impl` | **None** | `attn_implementation` for the transformers engine, e.g. `flash_attention_2`, `sdpa`, `eager`. None lets HF choose. |
| `--qwen-video-reader-backend` | **torchvision** | Forces the `qwen_omni_utils` video reader (`FORCE_QWENVL_VIDEO_READER`). `torchvision` avoids `torchcodec`, which often fails on mismatched CUDA/ffmpeg. Also `decord`, `torchcodec`. |

**Caption resume / cache** (qwen3_omni)

| Flag | Default | Meaning |
|------|---------|---------|
| `--resume` / `--no-resume` | **resume on** | Skip segments that already have a valid cached caption (no model call). `--no-resume` re-checks every segment. |
| `--overwrite-captions` | **off** | Force regeneration of every caption, ignoring any cache. |
| `--captions-cache-dir` | **`<output>/captions`** | Per-segment structured-caption cache root. Raw parse failures go to `<output>/captions_raw`. |

**Verify / rewrite backend**

| Flag | Default | Meaning |
|------|---------|---------|
| `--verify-rewrite-backend` | **qwen3_omni** | Backend for verification + rewrite (both watch the query's clip(s)). `qwen3_omni` watches local clips on the shared model (no upload); `gemini` uploads to the Files API. |
| `--max-rewrites` | **3** | Max rewrite rounds per failing query. |
| `--max-accepted` | **8** | Max accepted queries kept per video. |

**Gemini models** (only used for stages running on Gemini)

| Flag | Default | Meaning |
|------|---------|---------|
| `--generation-model` | **gemini-2.5-flash-lite** | Query-generation model (generation always runs on Gemini). |
| `--caption-model` | **gemini-2.5-flash-lite** | Caption model, used only when `--caption-backend gemini`. |
| `--verification-model` | **gemini-3.1-flash-lite** | Verifier model, used only when `--verify-rewrite-backend gemini`. |
| `--rewrite-model` | **gemini-2.5-flash-lite** | Rewrite model, used only when `--verify-rewrite-backend gemini`. |

**Transcript (WhisperX)**

| Flag | Default | Meaning |
|------|---------|---------|
| `--no-transcript` | **off (transcript on)** | Skip WhisperX; generation runs without dialogue text. |
| `--whisper-model` | **small** | WhisperX size (`tiny`, `base`, `small`, ...). |

**Default backend split:** caption + verify/rewrite run on **Qwen3-Omni**; generation
(query writing) runs on **Gemini**. `GEMINI_API_KEY` is therefore always required.

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
- **Two batching dimensions** (both default `1`):
  - `--caption-batch-size` = **segments per prompt**. The model sees N segment
    clips in one prompt (enumerated `Clip i -> segment_id`) and returns a JSON
    array of N captions, each mapped back to its `segment_id`. (This intentionally
    relaxes the original one-segment-per-prompt rule; a segment the model
    skips/garbles is raw-dumped and retried next run, never mixed.)
  - `--caption-parallel` = **prompts per `generate` call**. How many of those
    prompts run together in one batched model call for throughput.
  Resume is applied first, so only cache-miss segments are grouped. Larger values
  = fewer calls but more VRAM / a harder mapping for the model.
- **Structured output:** each caption is a nested JSON with `visual_objective`
  (objective facts only), `visual_expression` (observable facial/body/gaze cues),
  `audio_description` (non-transcript audio), `emotion_description` (a *candidate*
  reading — a full sentence), plus `confidence` / `evidence_strength`. **Generation
  consumes this rich structure directly** (the full `visual_objective` /
  `visual_expression` and the `emotion_description` sentence), so no information is
  lost to flattening. A flat `EmotionCaption` is still derived (via the
  `OmniCaption → EmotionCaption` adapter) only for `raw_captions.jsonl` and stats;
  its keyword-mapped `emotion` label is provenance, not what generation reads. The
  flat Gemini caption path maps into the same generation schema sparsely. There is
  no caption-filtering step — every caption is fed to generation, which selects
  moments.

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

The pure modules have no hard SDK imports and can be exercised directly:
`segmentation.plan_segments` / `grid_key`, `generation` (payload building for both
`OmniCaption` and `EmotionCaption`, and `_resolve_time_ranges`), and
`workflow.run_query_pipeline` (inject a fake `BaseLLMClient`). `llm_client` imports
`google.genai` lazily, so `BaseLLMClient` / `generation` import fine without it;
only constructing `GeminiLLMClient`, `GeminiUploader`, and `transcription`
(WhisperX) touch external models. See `tests/test_generation.py`.

The Qwen3-Omni backend is fully testable without the model — prompt construction,
JSON extraction, field validation, the cache/resume decision, atomic write and the
`OmniCaption → EmotionCaption` adapter are pure Python (heavy deps imported lazily):

```bash
python -m pytest tests/test_omni_captioning.py -q   # 22 tests, no GPU / no weights
```
