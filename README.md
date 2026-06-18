# v2 Caption-based Query Generation Pipeline

Generates emotion-related **temporal grounding queries** from raw videos by first
building per-segment **emotion captions**, then writing queries from the caption text.
Every query traces back `query → source_caption_id → segment_ids`, so grounding is tied
to specific time segments instead of the whole video.

This is a self-contained sibling of `video_query_answering_demo` (the v1 demo); it copies
the segmentation/clip-extraction utilities and the verify/rewrite stage rather than
importing them.

## Flow

```
raw .mp4
  └─ 1. ffprobe duration + cut fixed 5s segments (s001, s002, ...) with ffmpeg
  └─ 2. group segments into batches (default 8 = 40s)
  └─ 3. per batch: upload N clips → ONE multimodal LLM call → emotion captions
  └─ 4. filter: drop low-confidence / weak / ambiguous / ungrounded captions
  └─ 5. caption-only LLM call sees ALL of the video's captions, selects the
        strong ones → few, high-precision queries (+ source_caption_id, segment_ids)
  └─ 6. upload whole video once → verify ⇄ rewrite loop (prompts unchanged)
  └─ export intermediates + final queries, clean clips/uploads
```

Step 3 sends the actual clips (with audio); step 5 sends **only** the caption text (no
video) — the model reads every caption for the video and chooses which to ground queries
on. The eight emotion labels are fixed: `angry, excited, fear, sad, surprised,
frustrated, happy, disappointed`.

## Setup

```bash
pip install -r requirements.txt   # google-genai, pydantic>=2
export GEMINI_API_KEY="..."       # required
# ffmpeg + ffprobe must be on PATH (clip cutting + duration probing)
```

## Run

```bash
python run_pipeline.py \
  --video-dir "../video_query_answering_demo/data/pilot study" \
  --num-videos 1 \
  --output output/smoke
```

Key flags (all optional except `--video-dir`):
`--num-videos --seed --batch-size --segment-seconds --stride --max-rewrites --max-accepted`
`--caption-model --generation-model --verification-model --rewrite-model --temp-dir --keep-temp-clips`

Defaults: 5s segments / 5s stride (non-overlapping), batch 8, max_accepted 8, max_rewrites 3,
caption & generation `gemini-2.5-flash-lite`, verification `gemini-3.1-flash-lite`,
rewrite `gemini-2.5-flash-lite`.

## Outputs (under `--output`)

| File | Contents |
|------|----------|
| `segments.jsonl` | every segment with its time span |
| `raw_captions.jsonl` | captions straight from the model |
| `filtered_captions.jsonl` | after rule filtering (the captions generation reads) |
| `initial_queries.jsonl` | queries generated from captions |
| `verification_rounds.jsonl` | every verifier round |
| `rewritten_queries.jsonl` | every rewrite |
| `final_queries.jsonl` | per-query trace + final status (with provenance) |
| `human_review_sheet.csv` | accepted queries for human review |
| `pipeline_stats.json` | aggregate stats |
| `prompts_used/` | the 4 prompts + version manifest |

## Testing without API calls

The pure modules have no SDK imports and can be exercised directly:
`segmentation.plan_segments`, `caption_filter.filter_captions`, and
`workflow.run_query_pipeline` (inject a fake `BaseLLMClient`). Only `llm_client`
and `captioning.GeminiUploader` touch `google-genai`.
```
