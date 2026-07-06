"""Stage 3 of the caption-model plumbing check: verify/rewrite evaluation.

Reads a stage-2 output dir (``run_query_generation_test.py``: ``segments.jsonl``
+ ``generated_queries.json``), uploads each query's grounded clip(s) to Gemini,
and runs the pipeline's real verify<->rewrite loop
(``emotion_query_pipeline.workflow.run_query_pipeline``) — a ``pass`` accepts a
query as-is, ``revise`` applies its ``suggested_revision`` inline, ``fail`` drops
it. Needs ``GEMINI_API_KEY``.

Example:
    python run_evaluation_test.py \
        --queries-dir output/caption_query_tests/timechat \
        --output output/caption_query_tests/timechat
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from emotion_query_pipeline import caption_query_test as cqt


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run the verify/rewrite loop on cached generated queries "
        "(stage 3/3)."
    )
    p.add_argument(
        "--queries-dir", required=True,
        help="Stage-2 output dir (segments.jsonl + generated_queries.json).",
    )
    p.add_argument("--output", required=True, help="Output directory for evaluation artefacts.")
    p.add_argument("--verification-model", default=None)
    p.add_argument("--rewrite-model", default=None)
    return p


def _summarize(traces: list) -> dict:
    """Summarize a list of ``QueryTrace`` dicts from the verify<->rewrite loop.

    ``final_status`` is the query-level outcome (accepted/discarded);
    ``round_decisions`` breaks down every individual verification round's
    pass/revise/fail decision across all queries (a query can pass through
    several ``revise`` rounds before being accepted or eventually discarded).
    """
    final_status = Counter(t.get("final_status") for t in traces)
    round_decisions = Counter(
        r.get("decision") for t in traces for r in t.get("verification_rounds", [])
    )
    total = len(traces) or 1
    return {
        "total": len(traces),
        "final_status": dict(final_status),
        "final_status_pct": {
            k: round(100 * v / total, 1) for k, v in final_status.items()
        },
        "round_decisions": dict(round_decisions),
    }


def main() -> None:
    args = _build_arg_parser().parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    queries_dir = Path(args.queries_dir)
    segments, generation = cqt.load_generation_outputs(queries_dir)
    if not generation.queries:
        print(f"ERROR: no queries found in {queries_dir} "
              "(run run_query_generation_test.py first).", file=sys.stderr)
        sys.exit(2)
    video_id = generation.video_id
    print(f"[eval-test] video_id={video_id} segments={len(segments)} "
          f"queries={len(generation.queries)}")

    t0 = time.perf_counter()
    final_queries = cqt.run_verification_stage(
        video_id, generation, segments, api_key,
        verification_model=args.verification_model,
        rewrite_model=args.rewrite_model,
    )
    evaluation_seconds = time.perf_counter() - t0
    summary = _summarize(final_queries)
    print(f"[eval-test] {summary['total']} query(ies) checked in "
          f"{evaluation_seconds:.1f}s: "
          + ", ".join(f"{k}={v} ({summary['final_status_pct'][k]}%)"
                       for k, v in summary["final_status"].items()))

    metadata = {
        "video_id": video_id,
        "queries_dir": str(queries_dir),
        "num_queries_in": len(generation.queries),
        "num_queries_checked": summary["total"],
        "evaluation_seconds": round(evaluation_seconds, 1),
        "verification_model": args.verification_model,
        "rewrite_model": args.rewrite_model,
    }

    output_dir = Path(args.output)
    written = cqt.save_evaluation_outputs(
        output_dir, final_queries=final_queries, summary=summary, metadata=metadata,
    )
    print(f"\n[eval-test] wrote {len(written)} file(s) to {output_dir}:")
    for name in written:
        print(f"  - {name}")


if __name__ == "__main__":
    main()
