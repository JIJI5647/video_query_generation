# Dataset-generation pipeline — runbook (self-contained)

Run the full emotion-query dataset-generation pipeline on a folder of videos.
Everything needed is in `run_dataset_pipeline.sh`; this file explains it and lists
the gotchas so a fresh session can run it without prior context.

## What it does

For a folder of `.mp4` videos it runs, in order:

1. **Caption** — split each video into fixed segments; Qwen3-Omni-30B (served on
   vLLM) writes a structured audio-visual caption per segment.
2. **Emotion events** — Gemini reads the captions (text only) and groups them into
   emotion events (8 labels), cap on event length REMOVED.
3. **Query generation** — Gemini writes typed temporal queries from each event
   (with recurring-emotion disambiguation), each with a segment-level time span.
4. **Refine + guardrail** — Qwen3-Omni watches each query's clip and grounds it in
   observed detail; a guardrail reverts refinements that drop the emotion.
5. **Verify** — Qwen3-Omni scores each query on answerability / query-quality /
   emotion-relevance (variant p7_rolecot, per-dimension); passing queries are the
   dataset.

Human timestamp finalisation is a separate, later step (not in this script).

## One command

```bash
cd /work/mzha0323/video_query_generation
bash run_dataset_pipeline.sh <video-dir> <seg-seconds> <output-dir> [sample-N]
```

Recommended first run (50 videos from dev, 5-second segments — 5s is the chosen
primary granularity):

```bash
bash run_dataset_pipeline.sh data/gdrive_dataset/video_splits/dev 5 output/gen_dev50 50
```

- `sample-N` omitted or `0` = process ALL videos in the folder. The dataset splits
  live in `data/gdrive_dataset/video_splits/{dev,test,train}` (116 / 193 / 994
  videos). Do NOT run all 994 of `train` at once for a first pass — sample.
- The script is **resumable**: re-running skips any stage whose output already
  exists, and re-uses the same video selection (`<output-dir>/videos.sel`).

## Config baked in (the decisions already made)

- **Caption + verify + refine model:** Qwen3-Omni-30B-A3B-Instruct (local, served
  on vLLM). Captioning is fixed to Qwen3-Omni (highest caption quality).
- **Events + generation model:** Gemini (`gemini-2.5-flash`/`-flash-lite`), needs
  `GEMINI_API_KEY` (set in `env.sh`).
- **Segments:** 5 seconds is the primary choice (matches emotion time scale);
  2 seconds is finer but fragments more. The `<seg-seconds>` arg controls it.
- **Merge cap:** removed (long emotions span fully).

## GPU / serving

- One H200. The script serves Qwen3-Omni once (`run_vllm_serve_qwen.sh`, port
  8000, `GPU_UTIL=0.7`, `MAX_LEN=65536`) and reuses it for caption + refine +
  verify, then stops it on exit. Serve takes ~2–4 min to become ready.
- To stop a serve manually (leak-safe): kill the `vllm serve` API pid, then
  `pkill -TERM -f "VLLM::EngineCore"` (the renamed EngineCore child leaks GPU
  memory otherwise). The script's `stop_serve` does this.
- If GPU shows memory held with no process you recognise, it is almost always an
  orphaned `VLLM::EngineCore`; `pkill -TERM -f "VLLM::EngineCore"` frees it.

## Gotchas already handled in the script

- **`segments.jsonl` lacks `video_id`.** `run_caption_generation.py` omits it; the
  script patches it from `clip_path` right after captioning. (Without the patch,
  events/gen crash with `KeyError: 'video_id'`.)
- **vLLM captioning needs audio.** The served-caption route now passes
  `use_audio_in_video=True` (fixed in `caption_query_test.py`); this is what makes
  captioning ~15× faster than transformers while keeping audio evidence.
- **Qwen3-Omni Instruct has no thinking mode.** All calls pass `--thinking none` /
  `--nemotron-no-thinking`; do not send `enable_thinking=true`.

## Monitoring

- Progress: `tail -f logs/<output-basename>/caption.log` (or events/gen/refine/
  verify.log). The orchestrator prints `>>> [n/5] ... $(date)` markers to
  `logs/run_dataset_pipeline.log` if you redirect its stdout there.
- The whole run for ~50 short videos at 5s is roughly: caption ~10–20 min, events
  + gen ~10–20 min (Gemini), refine ~15–30 min, verify ~15–25 min.

## Output

`<output-dir>/final/initial_queries.jsonl` (the queries) and
`<output-dir>/final/verification_results.jsonl` (pass/fail per query). The script
prints a final summary: number of queries, pass count, pass rate.

## To export for human annotation

After a run, export pass queries with captions + empty human fields (mirrors
`output/compare_5s_19_all.json`): see `output/compare_2s_19_all.json` for the
schema, or ask the session to build the export from
`<output-dir>/final/` + the caption dir.
