# Caption → Emotion Event → Query Pipeline (v5)

Generates emotion-related **temporal grounding queries** from raw videos in three
separated stages:

1. **Observation captions** (no emotion): each segment is described purely by what
   is *visible* and *audible*.
2. **Emotion events** (the only place emotion is judged): a Gemini stage reads the
   observation captions and emits emotional moments, each labelled with one of
   eight emotion-relevant classes.
3. **Queries**: a Gemini stage writes time-grounded queries from the emotion
   events + observation captions, then a verify ⇄ rewrite loop filters/fixes them.

Each query is grounded to a **time range** `[start, end]` of the video. The eight
emotion labels are fixed: `angry, excited, fear, sad, surprised, frustrated,
happy, disappointed` (no `neutral`/`unrelevant` — a moment with no clear
emotion-relevant evidence simply produces no event).

This is a self-contained sibling of `video_query_answering_demo`; it copies the
segmentation/clip-extraction utilities and the verify/rewrite stage rather than
importing them.

## Flow

```
raw .mp4
  └─ 1. ffprobe duration + cut fixed 5s segments (s001, s002, ...) with ffmpeg,
        into a PERSISTENT cache: data/processed_segments/<video_id>/<grid_key>/
        (reused across runs; never auto-deleted)
  └─ 2. OBSERVATION captions (no emotion), one of three backends:
        • qwen3_vl_timechat (default): Qwen3-VL over N sampled frames (visual) +
                                       TimeChat over the clip (audio/temporal),
                                       merged per segment_id, cached for resume
        • qwen3_omni : legacy single Qwen3-Omni captioner (observation-only)
        • gemini     : Gemini Files API batch
  └─ 3. EMOTION EVENTS (Gemini, text-only): reads all observation captions →
        emotional moments, each with one of the 8 labels + observable evidence.
        This is the ONLY emotion judgment in the pipeline.
  └─ 4. QUERIES (Gemini, text-only): emotion events + observation captions →
        few, high-precision queries, each grounded on a time_range.
  └─ 5. for each query, take ONLY its grounded segment clip(s) → verify ⇄ rewrite
        (Gemini upload, or watched locally on Qwen3-Omni). The verifier sees ONLY
        query_id + query_text.
  └─ export intermediates + final queries; segment clips stay cached, uploads cleaned
```

Stage 2 sends the actual frames/clips; stages 3–4 send **only text** (no video).
There is no caption-filtering step — every observation caption feeds the
emotion-event stage. `segment_id` is an internal handle mapping a query's
`time_range` back to the clip(s) for verification — never shown to the models or to
human annotators.

Prompt versions: `observation_visual_prompt_v1`,
`observation_audio_temporal_prompt_v1`, `emotion_event_prompt_v1`,
`generation_prompt_events_v10`, `verification_prompt_v11`.

## Setup

```bash
pip install -r requirements.txt   # google-genai, pydantic>=2
export GEMINI_API_KEY="..."       # required (emotion-event + generation always on Gemini)
# ffmpeg + ffprobe must be on PATH (clip cutting, duration probing, frame sampling)
```

The local caption / verify backends additionally need `transformers`,
`qwen-vl-utils`, `qwen-omni-utils`, `torch`, `accelerate` (plus optional
`flash-attn`) — install these **only on the GPU inference server**, kept out of the
main requirements so the default tests stay light:

```bash
pip install -r requirements.txt -r requirements-qwen.txt   # GPU server only
```

Install the `torch` build matching your GPU driver's CUDA. Heavy deps and model
weights are imported/loaded lazily (only on the first inference call), so the rest
of the pipeline and all tests run without them — but a real run needs a GPU.

## Run

```bash
python -u run_pipeline.py \
  --video-dir data/pilot_study \
  --num-videos 5 \
  --output output/v5_run \
  --parallel 4
```

Selection: `--video-ids "id1,id2"` pins an exact set (file stems, in order) and
overrides the default `--num-videos`/`--seed` random sampling.

### Flags

All optional except `--video-dir`. Grouped by purpose; default in **bold**.

**Input / selection**

| Flag | Default | Meaning |
|------|---------|---------|
| `--video-dir` | *(required)* | Directory of source `.mp4` / `.avi` videos. |
| `--num-videos`, `-n` | **all** | How many videos to sample. Omit to process every video. |
| `--video-ids` | **None** | Comma/space-separated file stems to process exactly, in order. Overrides `--num-videos`/`--seed`. |
| `--seed` | **42** | Random seed for video sampling. |
| `--output` | **output/v2_run** | Output directory for all artefacts. |

**Segmentation / clips**

| Flag | Default | Meaning |
|------|---------|---------|
| `--segment-seconds` | **5.0** | Length of each fixed segment, in seconds. |
| `--stride` | **5.0** | Hop between segment starts (= `segment-seconds` → non-overlapping). |
| `--min-segment-seconds` | **1.0** | Drop a trailing segment shorter than this. |
| `--segments-dir` | **data/processed_segments** | Persistent segment-clip cache root (reused across runs). |
| `--force-reextract` | **off** | Re-cut segment clips even if cached. |
| `--temp-dir` / `--keep-temp-clips` | **temp_clips** / off | Scratch dir for clips; keep them instead of cleaning. |

**Caption backend (observation-only)**

| Flag | Default | Meaning |
|------|---------|---------|
| `--caption-backend` | **qwen3_vl_timechat** | `qwen3_vl_timechat` (Qwen3-VL frames + TimeChat clip, merged), `qwen3_omni` (legacy single model), or `gemini` (Files API). |
| `--caption-batch-size` | **1** | Segments per prompt (`qwen3_omni`/`gemini` only). |
| `--parallel` | **1** | How many prompts run in ONE batched `generate` call (throughput) — applies to captioning and verify/rewrite. Larger = more VRAM. Gemini runs sequentially. |
| `--frames-per-segment` | **5** | Frames sampled per segment for the Qwen3-VL visual backend. |
| `--qwen3vl-model-path` | **Qwen/Qwen3-VL-8B-Instruct** | Visual caption model. |
| `--timechat-model-path` | **yaolily/TimeChat-Captioner-GRPO-7B** | Audio/temporal caption model (`qwen2_5_omni` arch). |

**Legacy Qwen3-Omni model** (used by `--caption-backend qwen3_omni` and/or `--verify-rewrite-backend qwen3_omni`)

| Flag | Default | Meaning |
|------|---------|---------|
| `--qwen-model-path` | **Qwen/Qwen3-Omni-30B-A3B-Instruct** | HF model id / local path. |
| `--qwen-attn-impl` | **None** | `attn_implementation`, e.g. `flash_attention_2`, `sdpa`, `eager`. Shared by all transformers backends. |
| `--qwen-video-reader-backend` | **torchvision** | Forces the `qwen_omni_utils` video reader (`FORCE_QWENVL_VIDEO_READER`); avoids `torchcodec` on mismatched CUDA/ffmpeg. |

**Caption resume / cache**

| Flag | Default | Meaning |
|------|---------|---------|
| `--resume` / `--no-resume` | **resume on** | Skip segments with a valid cached caption (no model call). |
| `--overwrite-captions` | **off** | Force regeneration of every caption, ignoring any cache. |
| `--captions-cache-dir` | **`<output>/captions`** | Per-segment caption cache root. Raw parse failures go to `<output>/captions_raw`. |

**Emotion-event + query + verify/rewrite**

| Flag | Default | Meaning |
|------|---------|---------|
| `--emotion-event-model` | **= generation model** | Gemini model for the emotion-event stage. |
| `--verify-rewrite-backend` | **qwen3_omni** | `qwen3_omni` watches local clips on the shared model (no upload); `gemini` uploads to the Files API. |
| `--max-rewrites` | **3** | Max rewrite rounds per query needing revision. |
| `--max-accepted` | **8** | Max accepted queries kept per video. |

**Gemini models** (for stages running on Gemini)

| Flag | Default | Meaning |
|------|---------|---------|
| `--generation-model` | **gemini-2.5-flash-lite** | Query-generation model (always on Gemini). |
| `--caption-model` | **gemini-2.5-flash-lite** | Used only when `--caption-backend gemini`. |
| `--verification-model` | **gemini-3.1-flash-lite** | Used only when `--verify-rewrite-backend gemini`. |
| `--rewrite-model` | **gemini-2.5-flash-lite** | Used only when `--verify-rewrite-backend gemini`. |

**Default backend split:** captioning runs on **Qwen3-VL + TimeChat**;
emotion-event + query generation run on **Gemini**; verify/rewrite runs on
**Qwen3-Omni**. `GEMINI_API_KEY` is always required.

> **VRAM note:** the default puts three local models in play — Qwen3-VL-8B +
> TimeChat-7B (captioning) and Qwen3-Omni-30B (verify/rewrite). If the box OOMs,
> move verify/rewrite to Gemini with `--verify-rewrite-backend gemini`.

### Caption backends

The default **`qwen3_vl_timechat`** produces observation-only captions from two
models, merged per `segment_id`:

- **Qwen3-VL** reads `--frames-per-segment` sampled frames and fills the VISUAL
  fields: `visual_objective` (objective facts) + `visual_expression` (observable
  facial/body/gaze cues) + `confidence`/`evidence_strength`.
- **TimeChat** (`qwen2_5_omni` arch) reads the segment clip and fills
  `audio_description` (non-verbal audio) + `temporal_description`. No transcript,
  no emotion.

Both load lazily (no GPU touched at import/construct time). A missing/garbled half
is salvaged so a segment is never dropped; the raw output is dumped to
`<output>/captions_raw/` for debugging.

`--caption-backend qwen3_omni` keeps the legacy single-model observation captioner;
`--caption-backend gemini` uses the Files API batch path.

**Resume / cache:** each observation caption is written atomically to
`<output>/captions/<video_id>/<segment_id>.json`. On rerun, a segment with a valid
cached caption is **skipped without calling the model** (`--resume`, default on).
`--overwrite-captions` forces regeneration; salvaged captions are left uncached so
a later run retries them.

### Emotion-event stage

A text-only Gemini call reads all of a video's observation captions and emits
`EmotionEvent`s — the **only** emotion judgment in the pipeline. Each event has one
of the eight labels, a `time_range` (resolved internally to `segment_ids`), a
`target_person_or_group`, and the `visual_evidence`/`audio_evidence` behind the
label. Moments without clear emotion-relevant evidence yield no event.

### Segment cache

Segment clips are cut once into `data/processed_segments/<video_id>/<grid_key>/`,
where `grid_key` (e.g. `win5.00_str5.00`) encodes the windowing params. Reruns
reuse existing clips unless `--force-reextract`.

## Outputs (under `--output`)

| File | Contents |
|------|----------|
| `segments.jsonl` | every segment with its time span |
| `raw_captions.jsonl` | **observation** captions (no emotion), all fed downstream |
| `emotion_events.jsonl` | emotion events (8 labels) from the emotion-event stage |
| `initial_queries.jsonl` | queries generated from events + captions |
| `verification_rounds.jsonl` | every verifier round |
| `rewritten_queries.jsonl` | every applied revision |
| `final_queries.jsonl` | per-query trace + final status; `time_range` (external) + `segment_ids` (internal) |
| `human_review_sheet.csv` | accepted queries for review; `time_range` column, no segment ids |
| `pipeline_stats.json` | aggregate stats (emotion distribution now from events) |
| `prompts_used/` | the prompt templates + version manifest |
| `captions/<video_id>/<segment_id>.json` | observation caption resume cache |
| `captions_raw/<video_id>/<segment_id>.txt` | raw model text saved on a caption parse failure |

`rerun_generation.py` re-runs only the emotion-event + query + verify/rewrite
stages from a prior run's observation captions (reusing the segment cache).

## Testing without API calls

The pure modules have no hard SDK / model imports and can be exercised directly:
`segmentation.plan_segments` / `grid_key`; `generation` (observation payload for
`OmniCaption` and `EmotionCaption`, `_resolve_time_ranges`); `emotion_events`
(time-range resolution); `observation_captioning.caption_video_observation` (merge
+ cache, with fake captioners); and `workflow.run_query_pipeline` (fake
`BaseLLMClient`). Heavy deps (`torch`, `transformers`, `qwen_vl_utils`,
`qwen_omni_utils`) and `google.genai` are imported lazily.

```bash
python -m pytest tests/ -q   # no GPU, no weights, no network
```
