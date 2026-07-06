"""Stage 2 of the caption-model plumbing check: Gemini query generation.

Reads a stage-1 output dir (``run_caption_generation_test.py``: ``segments.jsonl``
+ ``normalized_captions.jsonl``) and runs the REAL Gemini emotion-event +
query-generation stages, unmodified. Needs ``GEMINI_API_KEY``; no GPU, no caption
model. See ``run_evaluation_test.py`` for the third stage (verify/rewrite).

Example:
    python run_query_generation_test.py \
        --captions-dir output/caption_query_tests/timechat \
        --output output/caption_query_tests/timechat
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from emotion_query_pipeline import caption_query_test as cqt


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run Gemini emotion-event + query-generation on cached "
        "captions (stage 2/3)."
    )
    p.add_argument(
        "--captions-dir", required=True,
        help="Stage-1 output dir (segments.jsonl + normalized_captions.jsonl).",
    )
    p.add_argument("--output", required=True, help="Output directory for query artefacts.")
    p.add_argument("--generation-model", default="gemini-2.5-flash-lite")
    p.add_argument(
        "--emotion-event-model", default=None,
        help="Defaults to the generation model.",
    )
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    captions_dir = Path(args.captions_dir)
    segments, captions = cqt.load_caption_outputs(captions_dir)
    if not segments or not captions:
        print(f"ERROR: no segments/captions found in {captions_dir} "
              "(run run_caption_generation_test.py first).", file=sys.stderr)
        sys.exit(2)
    video_id = captions[0].video_id
    print(f"[query-gen-test] video_id={video_id} segments={len(segments)} "
          f"captions={len(captions)}")

    from emotion_query_pipeline.llm_client import GeminiLLMClient
    client = GeminiLLMClient(
        generation_model=args.generation_model,
        emotion_event_model=args.emotion_event_model or args.generation_model,
        api_key=api_key,
    )
    inputs = cqt.build_downstream_inputs(video_id, captions, segments)
    t0 = time.perf_counter()
    downstream = cqt.run_downstream_gemini(inputs, client)
    generation_seconds = time.perf_counter() - t0
    events, generation = downstream["events"], downstream["generation"]
    warnings = list(downstream["warnings"])
    print(f"[query-gen-test] {len(events.events)} emotion event(s), "
          f"{len(generation.queries)} query(ies) in {generation_seconds:.1f}s")
    for w in warnings:
        print(f"  WARNING: {w}")

    metadata = {
        "video_id": video_id,
        "captions_dir": str(captions_dir),
        "num_segments": len(segments),
        "num_emotion_events": len(events.events),
        "num_generated_queries": len(generation.queries),
        "generation_seconds": round(generation_seconds, 1),
        "generation_model": args.generation_model,
        "emotion_event_model": args.emotion_event_model or args.generation_model,
        "warnings": warnings,
    }

    output_dir = Path(args.output)
    written = cqt.save_generation_outputs(
        output_dir, events=events, generation=generation, segments=segments,
        metadata=metadata,
    )
    print(f"\n[query-gen-test] wrote {len(written)} file(s) to {output_dir}:")
    for name in written:
        print(f"  - {name}")
    if not generation.queries:
        print("\n[query-gen-test] NOTE: 0 queries generated "
              "(see generation_metadata.json warnings).")
    else:
        print(f"\n[query-gen-test] next: python run_evaluation_test.py "
              f"--queries-dir {output_dir} --output <eval-output-dir>")


if __name__ == "__main__":
    main()
