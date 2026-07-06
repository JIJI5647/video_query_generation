# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Generates emotion-related **temporal grounding queries** from raw videos: per-5s-segment
emotion captions are built first, then queries are written from the caption text. Every
query is grounded to a `time_range` of the video rather than the whole clip. This is a
self-contained sibling of `video_query_answering_demo` (the v1 demo) — it copies
segmentation/clip-extraction utilities and the verify/rewrite stage rather than importing
them.

Pipeline: `raw .mp4 → segment+cut clips → caption each segment →
caption-only query generation (Gemini) → per-query verify⇄rewrite loop (watches only that
query's grounded clip(s)) → export`. Full details, all CLI flags, and output file formats
are in `README.md` — read it before changing `run_pipeline.py` or the caption/verify
backends. (There is no transcript stage — WhisperX support was removed; `generation.py`
runs on caption text only.)

## Commands

```bash
# Tests — pure-Python, no GPU/API key, run these before/after any change
python -m pytest tests/ -q
python -m pytest tests/test_omni_captioning.py -q       # Qwen3-Omni backend (prompt/JSON/cache logic)
python -m pytest tests/test_generation.py -q            # query-generation payload building (fake LLM client)
python -m pytest tests/test_verification_perdim.py -q   # per-dimension verification composer/routing
python -m pytest tests/test_caption_query_utils.py -q   # caption-query-test stage scripts' normalization/save/load helpers
python -m pytest tests/test_caption_query_integration.py -q   # SKIPPED unless RUN_CAPTION_QUERY_INTEGRATION=1 (needs GPU + GEMINI_API_KEY)
python -m pytest path/to/test_file.py::test_name -q     # single test

# Full pipeline (needs GEMINI_API_KEY; caption+verify/rewrite default to local Qwen3-Omni, generation always Gemini)
python run_pipeline.py --video-dir data/pilot_study --video-ids "id1,id2" --output output/run_name

# Re-run only generation+verification from a prior run's cached captions
python rerun_generation.py --captions-dir output/prev_run --video-dir data/pilot_study --output output/regen

# Re-run only verification (single pass, no rewrite loop) — for iterating on the verification prompt/model
python run_verification.py --queries-dir output/prev_run --video-dir data/pilot_study --output output/prev_run_verify

# Merge several run output dirs into one (dedup by video_id, later dirs win)
python merge_runs.py --into output/merged --from output/run_a output/run_b

# Score verification against human-annotated gold labels
python eval_verification.py --gold data/eval/gold.jsonl --results output/verify_*/verification_results.jsonl

# New caption-model plumbing check (caption model -> real Gemini downstream; see docs/caption_query_test.md)
# 3 independent stages — each cacheable/re-runnable on its own:
python run_caption_generation_test.py --caption-model qwen3_omni --video short.mp4 --output output/caption_query_tests/qwen3_omni --max-segments 3
python run_query_generation_test.py --captions-dir output/caption_query_tests/qwen3_omni --output output/caption_query_tests/qwen3_omni
python run_evaluation_test.py --queries-dir output/caption_query_tests/qwen3_omni --output output/caption_query_tests/qwen3_omni
```

Setup: `pip install -r requirements.txt` (light: `google-genai`, `pydantic>=2`); add
`-r requirements-qwen.txt` **only on the GPU inference server** for the Qwen3-Omni/Qwen3-VL
backends (heavy deps — torch/transformers/qwen-omni-utils/qwen-vl-utils — are imported
lazily, so the rest of the pipeline and all tests run without them). `ffmpeg`/`ffprobe` must
be on PATH. `GEMINI_API_KEY` is required for any run that touches generation (always) or a
`gemini` backend for caption/verify.

**Resource limits on this machine: 8 CPU cores / 32GB RAM available (not the host's full
224-core/2TB — those are visible via `nproc`/`free -h` but not actually usable).** Keep
`--parallel`, batch sizes, and worker counts modest accordingly; the GPU (1x H200, 143GB) is
not subject to this limit. Check `/sys/fs/cgroup/memory.current` vs `memory.max` for real
usage, not `free -h`.

**This server reclaims the machine if the GPU sits at 0% utilization for 30 minutes** —
`gpu_keepalive.sh` runs a small `torch` matmul burst whenever it polls (`nvidia-smi`, every
5 min) and finds all GPUs idle, to keep utilization above 0. Start it whenever a session begins
GPU-idle work (e.g. writing/reviewing code, not actively running inference/training):

```bash
nohup bash gpu_keepalive.sh > /dev/null 2>&1 &
disown
# stop:  kill $(cat gpu_keepalive.pid)
# check it's alive: ps -p $(cat gpu_keepalive.pid)
```

It writes its own PID to `gpu_keepalive.pid` and logs each check to `gpu_keepalive.log`. This
PID file can go stale (process died but file remains) — always verify with `ps -p $(cat
gpu_keepalive.pid)` before assuming it's protecting the session, don't trust the file's mere
existence.

## Architecture

All pipeline logic lives in the `emotion_query_pipeline/` package; the top-level `run_*.py` /
`rerun_generation.py` / `merge_runs.py` scripts are thin CLI entry points that wire its
modules together for different partial re-runs. Key modules:

- **`models.py`** — every Pydantic schema (`Segment`, `EmotionCaption`, `OmniCaption`,
  `EventGroundedQuery`, `VerificationResult`, `QueryTrace`, …). The eight fixed emotion
  labels (`angry, excited, fear, sad, surprised, frustrated, happy, disappointed`) are
  defined once here (`EMOTION_LABEL_VALUES`); caption-stage labels add `neutral`/`unrelevant`,
  event-stage labels do not.
- **`segmentation.py`** — fixed-window segment planning (`plan_segments`, `grid_key`) and
  clip cutting into the **persistent** cache `data/processed_segments/<video_id>/<grid_key>/`
  (never auto-deleted; a different segment-length/stride writes a fresh subdir via
  `grid_key`, e.g. `win5.00_str5.00`).
- **`captioning.py`** / **`omni_captioning.py`** — the two pluggable caption backends:
  Gemini (Files API, batch upload) and Qwen3-Omni (local `transformers` engine, lazy-loaded
  on first inference call). Both batch on two independent axes: segments-per-prompt
  (`--caption-batch-size`) and prompts-per-`generate()`-call (`--parallel`). Qwen3-Omni
  writes structured per-segment JSON captions atomically to `<output>/captions/` as a resume
  cache (parse failures dump raw text to `captions_raw/` for debugging).
- **`generation.py`** — the caption-only query-generation stage: sees ALL of a video's
  captions (with time ranges), no video and no transcript (WhisperX support was removed).
  Consumes the Qwen3-Omni structured fields (`visual_objective`, `visual_expression`,
  `audio_description`, `emotion_description`) directly rather than a flattened label.
- **`emotion_events.py`** — intermediate stage between captions and generation that groups
  captions into discrete emotion events.
- **`verification.py`** / **`workflow.py`** — the combined verify⇄rewrite loop.
  Verification and rewriting are **one call**: a `revise` verdict already carries a concrete
  `suggested_revision` that is applied inline (no separate rewrite call); `pass` accepts,
  `fail` discards immediately (never revised). Each query is checked against ONLY the clip(s)
  of the segments it's grounded on, not the whole video; the verifier sees only
  `query_id` + `query_text`, never caption metadata. `verify_queries_per_dimension` (used by
  the perdim ablation, see below) judges relevance/query_quality from query text alone and
  answerability from the clip.
- **`llm_client.py`** — `BaseLLMClient` abstraction; `GeminiLLMClient` and
  `QwenOmniLLMClient` implementations. `google.genai` is imported lazily so the rest of the
  package (and all tests) import fine without the SDK installed.
- **`export.py`** / **`stats.py`** / **`validation.py`** — write the final
  `output/<run>/*.jsonl` artefacts (see the Outputs table in `README.md`), compute aggregate
  stats, and validate cross-file consistency.
- **`clip_extractor.py`** / **`video_utils.py`** / **`io_utils.py`** — ffmpeg clip
  cutting/probing and shared file/prompt-template loading helpers.

**`run_caption_generation_test.py`** / **`run_query_generation_test.py`** /
**`run_evaluation_test.py`** are a separate plumbing-check tool (not part of the main
pipeline) for testing whether a *new* caption model can feed the existing Gemini downstream —
see `docs/caption_query_test.md` for the full model matrix (`qwen3_omni`, `qwen_audio_vl`,
`af3_vl`, `secap_qwen`, `avocado`, `timechat` — several are stubs that raise a clear
`NotImplementedError`). Split into 3 independent, cacheable stages (each reads the prior
stage's output dir): caption generation (normalizes any caption model's raw output into the
pipeline's `OmniCaption` schema via `normalize_to_omni_caption` in
`emotion_query_pipeline/caption_query_test.py`) → Gemini emotion-event + query-generation →
verify/rewrite evaluation. This is an integration/plumbing check only, not a caption-quality
or query-quality evaluation.

### Prompts (`prompts/`)

All prompt files are **flat in this directory** — code loads them by bare filename and
`{{include: file.txt}}` directives (resolved in `io_utils.load_prompt_template`) inline
shared rule fragments at load time, so moving a file breaks every prompt that includes it.
`prompts/perdim/` holds the verification-ablation experiment: each `vdim_<variant>_<dim>.txt`
composes one (strategy variant p0-p8) x (dimension: relevance/answerability/query_quality)
combination via `{{include}}`; the experiment design lives in these files, not in Python. See
`prompts/README.md` for the full ablation methodology and `run_verification_sweep.sh` /
`eval_verification.py` for how variants are run and scored.

### Testing without API calls / GPU

Pure modules have no hard SDK imports and are directly testable: `segmentation.plan_segments`
/ `grid_key`, `generation` (payload building for `OmniCaption` and `EmotionCaption`, and
`_resolve_time_ranges`), `workflow.run_query_pipeline` (inject a fake `BaseLLMClient`), and
the entire Qwen3-Omni backend — prompt construction, JSON extraction, field validation, the
cache/resume decision, atomic write, and the `OmniCaption → EmotionCaption` adapter are pure
Python (heavy deps imported lazily). Only constructing `GeminiLLMClient` or
`GeminiUploader` touches external services/models.
