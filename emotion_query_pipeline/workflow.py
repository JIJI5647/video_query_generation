"""Verify -> rewrite loop over caption-generated queries (whole video).

Generation is done upstream (caption-based, no video); this module takes the
resulting ``GenerationOutput`` plus a whole-video URI and runs the same
verify -> rewrite -> re-verify loop as v1. Query grounding (``segment_ids``) is
preserved through traces and rewrites. The accepted cap is a parameter
(``max_accepted``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .llm_client import BaseLLMClient
from .models import (
    EmotionCaption,
    EventGroundedQuery,
    GenerationOutput,
    QueryTrace,
    RewriteBatchOutput,
    RoundDecision,
    Segment,
    VerificationBatchOutput,
    VerificationResult,
)
from .rewriting import rewrite_queries
from .verification import verify_queries


@dataclass
class PipelineResult:
    """All artefacts produced across the whole run (keyed by video_id)."""

    video_traces: Dict[str, Dict[str, QueryTrace]] = field(default_factory=dict)
    gen_outputs: Dict[str, GenerationOutput] = field(default_factory=dict)
    ver_outputs: Dict[str, List[VerificationBatchOutput]] = field(default_factory=dict)
    rw_outputs: Dict[str, List[RewriteBatchOutput]] = field(default_factory=dict)
    segments: Dict[str, List[Segment]] = field(default_factory=dict)
    raw_captions: Dict[str, List[EmotionCaption]] = field(default_factory=dict)
    filtered_captions: Dict[str, List[EmotionCaption]] = field(default_factory=dict)
    validation_warnings: List[str] = field(default_factory=list)


def _make_trace(q: EventGroundedQuery) -> QueryTrace:
    return QueryTrace(
        video_id=q.video_id,
        query_id=q.query_id,
        initial_query=q,
        current_query_text=q.query_text,
        final_query_text="",
        query_type=q.query_type,
        grounding_event_description=q.grounding_event_description,
        approximate_grounding_time=q.approximate_grounding_time,
        target_person_or_group=q.target_person_or_group,
        expected_evidence=list(q.expected_evidence),
        segment_ids=list(q.segment_ids),
        rewrite_count=0,
        verification_rounds=[],
        final_status="discarded",
    )


def _apply_verification_results(
    results: List[VerificationResult],
    traces: Dict[str, QueryTrace],
    pending: set,
    round_index: int,
) -> None:
    for result in results:
        qid = result.query_id
        if qid not in traces:
            continue
        traces[qid].verification_rounds.append(
            RoundDecision(
                round_index=round_index,
                decision=result.decision,
                failure_reason=result.failure_reason,
            )
        )
        if result.decision == "pass":
            traces[qid].final_status = "accepted"
            traces[qid].final_query_text = traces[qid].current_query_text
            pending.discard(qid)


def _apply_rewrites(
    rw_output: RewriteBatchOutput,
    traces: Dict[str, QueryTrace],
    current_queries: Dict[str, EventGroundedQuery],
) -> None:
    for rw in rw_output.rewrites:
        qid = rw.query_id
        if qid not in traces:
            continue
        traces[qid].rewrite_count += 1
        traces[qid].current_query_text = rw.rewritten_query_text
        traces[qid].query_type = rw.query_type
        old_q = current_queries[qid]
        current_queries[qid] = EventGroundedQuery(
            video_id=old_q.video_id,
            query_id=old_q.query_id,
            query_type=rw.query_type,
            query_text=rw.rewritten_query_text,
            grounding_event_description=old_q.grounding_event_description,
            approximate_grounding_time=old_q.approximate_grounding_time,
            target_person_or_group=old_q.target_person_or_group,
            expected_evidence=list(old_q.expected_evidence),
            why_grounded=old_q.why_grounded,
            segment_ids=list(old_q.segment_ids),
        )


def run_query_pipeline(
    video_id: str,
    video_uri: str,
    gen_output: GenerationOutput,
    client: BaseLLMClient,
    max_rewrites: int = 3,
    max_accepted: int = 8,
    prompts_dir: Optional[Path] = None,
) -> Tuple[
    Dict[str, QueryTrace],
    List[VerificationBatchOutput],
    List[RewriteBatchOutput],
]:
    """Run the verify -> rewrite loop for one video's generated queries.

    ``video_uri`` is the whole-video Files API URI (verification/rewrite watch
    the full video, exactly as in v1).
    """
    traces: Dict[str, QueryTrace] = {}
    current_queries: Dict[str, EventGroundedQuery] = {}
    for q in gen_output.queries:
        traces[q.query_id] = _make_trace(q)
        current_queries[q.query_id] = q

    pending: set = set(current_queries.keys())
    all_ver: List[VerificationBatchOutput] = []
    all_rw: List[RewriteBatchOutput] = []

    # Initial verification (round 1)
    round_index = 1
    ver_output = verify_queries(
        video_id, video_uri,
        [current_queries[qid] for qid in pending],
        round_index, client, prompts_dir,
    )
    all_ver.append(ver_output)
    last_ver_results: Dict[str, VerificationResult] = {
        r.query_id: r for r in ver_output.results
    }
    _apply_verification_results(ver_output.results, traces, pending, round_index)

    # Rewrite loop
    for rewrite_round in range(1, max_rewrites + 1):
        if not pending:
            break
        failing: List[Tuple[EventGroundedQuery, VerificationResult]] = [
            (current_queries[qid], last_ver_results[qid])
            for qid in pending
            if qid in last_ver_results
        ]
        if not failing:
            break

        rw_output = rewrite_queries(
            video_id, video_uri, failing, rewrite_round, client, prompts_dir
        )
        all_rw.append(rw_output)
        _apply_rewrites(rw_output, traces, current_queries)

        round_index += 1
        ver_output = verify_queries(
            video_id, video_uri,
            [current_queries[qid] for qid in pending],
            round_index, client, prompts_dir,
        )
        all_ver.append(ver_output)
        last_ver_results = {r.query_id: r for r in ver_output.results}
        _apply_verification_results(ver_output.results, traces, pending, round_index)

    # Discard remaining unresolved queries
    for qid in pending:
        traces[qid].final_status = "discarded"
        traces[qid].final_query_text = traces[qid].current_query_text

    # Configurable accepted cap (keep the lowest query_ids deterministically)
    accepted = [t for t in traces.values() if t.final_status == "accepted"]
    if len(accepted) > max_accepted:
        for t in sorted(accepted, key=lambda x: x.query_id)[max_accepted:]:
            t.final_status = "discarded"

    return traces, all_ver, all_rw
