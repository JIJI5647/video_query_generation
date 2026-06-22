"""Merge one or more v2 run output dirs into a single consolidated output dir.

Loads each dir's JSONL artefacts back into a ``PipelineResult`` (every record was
produced by ``model_dump`` so it round-trips), deduplicates by ``video_id``
(later dirs win), then recomputes stats + validation warnings and re-exports the
consolidated set with ``export_all``.

Usage:
    python merge_runs.py --into output/pilot_study \
        --from output/pilot_study output/pilot_study_retry
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from emotion_query_pipeline.export import export_all
from emotion_query_pipeline.models import (
    EmotionCaption,
    EventGroundedQuery,
    GenerationOutput,
    QueryTrace,
    RewriteBatchOutput,
    RoundDecision,
    Segment,
    VerificationBatchOutput,
)
from emotion_query_pipeline.stats import compute_stats
from emotion_query_pipeline.validation import validate_all
from emotion_query_pipeline.workflow import PipelineResult


def _read_jsonl(path: Path) -> List[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _trace_from_final_record(rec: dict) -> QueryTrace:
    initial_query = EventGroundedQuery(
        video_id=rec["video_id"],
        query_id=rec["query_id"],
        query_type=rec["query_type"],
        query_text=rec["initial_query_text"],
        grounding_event_description=rec["grounding_event_description"],
        approximate_grounding_time=rec.get("approximate_grounding_time"),
        target_person_or_group=rec["target_person_or_group"],
        expected_evidence=rec.get("expected_evidence", []),
        segment_ids=rec.get("segment_ids", []),
    )
    return QueryTrace(
        video_id=rec["video_id"],
        query_id=rec["query_id"],
        initial_query=initial_query,
        current_query_text=rec["final_query_text"] or rec["initial_query_text"],
        final_query_text=rec["final_query_text"],
        query_type=rec["query_type"],
        grounding_event_description=rec["grounding_event_description"],
        approximate_grounding_time=rec.get("approximate_grounding_time"),
        target_person_or_group=rec["target_person_or_group"],
        expected_evidence=rec.get("expected_evidence", []),
        segment_ids=rec.get("segment_ids", []),
        rewrite_count=rec.get("rewrite_count", 0),
        verification_rounds=[RoundDecision(**rd) for rd in rec.get("verification_rounds", [])],
        final_status=rec["final_status"],
    )


def _load_dir(result: PipelineResult, d: Path) -> None:
    """Load one run dir into ``result``, overwriting any existing video_ids."""
    # Caption-stage intermediates (group by video_id)
    seg_by_v: Dict[str, List[Segment]] = {}
    for r in _read_jsonl(d / "segments.jsonl"):
        seg_by_v.setdefault(r["video_id"], []).append(Segment.model_validate(r))
    for vid, segs in seg_by_v.items():
        result.segments[vid] = segs

    by_v: Dict[str, List[EmotionCaption]] = {}
    for r in _read_jsonl(d / "raw_captions.jsonl"):
        by_v.setdefault(r["video_id"], []).append(EmotionCaption.model_validate(r))
    for vid, caps in by_v.items():
        result.raw_captions[vid] = caps

    # Query-stage
    for r in _read_jsonl(d / "initial_queries.jsonl"):
        go = GenerationOutput.model_validate(r)
        result.gen_outputs[go.video_id] = go

    ver_by_v: Dict[str, List[VerificationBatchOutput]] = {}
    for r in _read_jsonl(d / "verification_rounds.jsonl"):
        vb = VerificationBatchOutput.model_validate(r)
        ver_by_v.setdefault(vb.video_id, []).append(vb)
    for vid, vbs in ver_by_v.items():
        result.ver_outputs[vid] = vbs

    rw_by_v: Dict[str, List[RewriteBatchOutput]] = {}
    for r in _read_jsonl(d / "rewritten_queries.jsonl"):
        rb = RewriteBatchOutput.model_validate(r)
        rw_by_v.setdefault(rb.video_id, []).append(rb)
    for vid, rbs in rw_by_v.items():
        result.rw_outputs[vid] = rbs

    traces_by_v: Dict[str, Dict[str, QueryTrace]] = {}
    for r in _read_jsonl(d / "final_queries.jsonl"):
        t = _trace_from_final_record(r)
        traces_by_v.setdefault(t.video_id, {})[t.query_id] = t
    for vid, traces in traces_by_v.items():
        result.video_traces[vid] = traces


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--into", required=True, help="output dir to write the merged result")
    ap.add_argument("--from", dest="sources", nargs="+", required=True,
                    help="source dirs in priority order (later wins on video_id)")
    ap.add_argument("--max-accepted", type=int, default=8)
    args = ap.parse_args()

    result = PipelineResult()
    for src in args.sources:
        _load_dir(result, Path(src))

    warnings = validate_all(result.video_traces, max_accepted=args.max_accepted)
    result.validation_warnings = warnings
    stats = compute_stats(
        result.video_traces,
        result.gen_outputs,
        result.ver_outputs,
        result.segments,
        result.raw_captions,
        warnings,
    )

    out = Path(args.into)
    export_all(result, out, stats)
    print(f"Merged {len(result.video_traces)} videos into {out}")
    print(f"  segments={stats.total_segments} captions={stats.total_raw_captions}")
    print(f"  initial={stats.total_initial_queries} accepted={stats.total_accepted_queries} "
          f"discarded={stats.total_discarded_queries}")


if __name__ == "__main__":
    main()
