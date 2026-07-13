"""Combined verify+revise loop over caption-generated queries (per-segment).

Generation is done upstream (caption-based, no video); this module takes the
resulting ``GenerationOutput`` plus a ``segment_id -> clip URI`` map and runs a
single combined loop. Verification defaults to the per-dimension architecture
(``verify_queries_per_dimension``, variant ``p7_rolecot`` — role framing + CoT,
each of relevance/answerability/query_quality judged in its own call; see
``prompts/README.md``). That path never returns a ``suggested_revision`` (only
the combined single-call ``verify_queries``/``verify_queries_many`` — still used
by ``run_verification.py``'s non-per-dimension arm — does), so:

  * ``pass``   -> accepted as-is;
  * ``fail``   -> discarded immediately (never revised);
  * ``revise`` -> no suggestion to apply inline, so it is discarded too — this
                  variant relies entirely on the generation stage's initial
                  quality, not on a rewrite loop. ``max_rewrites`` is kept for
                  the (rare) case a caller wires up a combined-call client.

Each query is verified while the model watches ONLY the clips of the segments the
query is grounded on (``segment_ids``) -- not the whole video. Calls are per-query;
per-round outputs are merged so downstream stats/export are unchanged (we still
emit ``RewriteRecord``s for every applied revision, when there are any). A
per-query API failure discards just that query instead of aborting the whole
video. The accepted cap is a parameter (``max_accepted``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .llm_client import BaseLLMClient
from .models import (
    EmotionCaption,
    EmotionEventOutput,
    EventGroundedQuery,
    GenerationOutput,
    QueryTrace,
    RewriteBatchOutput,
    RewriteRecord,
    RoundDecision,
    Segment,
    VerificationBatchOutput,
    VerificationResult,
)
from .verification import verify_queries, verify_queries_many, verify_queries_per_dimension


@dataclass
class PipelineResult:
    """All artefacts produced across the whole run (keyed by video_id)."""

    video_traces: Dict[str, Dict[str, QueryTrace]] = field(default_factory=dict)
    gen_outputs: Dict[str, GenerationOutput] = field(default_factory=dict)
    ver_outputs: Dict[str, List[VerificationBatchOutput]] = field(default_factory=dict)
    rw_outputs: Dict[str, List[RewriteBatchOutput]] = field(default_factory=dict)
    segments: Dict[str, List[Segment]] = field(default_factory=dict)
    raw_captions: Dict[str, list] = field(default_factory=dict)
    emotion_events: Dict[str, EmotionEventOutput] = field(default_factory=dict)
    validation_warnings: List[str] = field(default_factory=list)
    # Re-grounding stage stats per video: {"total": N, "changed": C, "fallback": F}
    # (see ``regrounding.reground_queries``). Empty for a video where
    # regrounding was disabled or produced no queries to reground.
    regrounding: Dict[str, Dict[str, int]] = field(default_factory=dict)


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
        time_range=list(q.time_range) if q.time_range else None,
        segment_ids=list(q.segment_ids),
        grounding_evidence=q.grounding_evidence,
        source_caption_ids=list(q.source_caption_ids),
        gen_time_range=list(q.gen_time_range),
        gen_segment_ids=list(q.gen_segment_ids),
        rewrite_count=0,
        verification_rounds=[],
        final_status="discarded",
    )


def _apply_verification_results(
    results: List[VerificationResult],
    traces: Dict[str, QueryTrace],
    pending: set,
    current_queries: Dict[str, EventGroundedQuery],
    round_index: int,
    max_rewrites: int,
) -> List[RewriteRecord]:
    """Route each verification decision and apply inline revisions.

    - ``pass``   -> accepted, leaves the pending set (done).
    - ``fail``   -> discarded immediately, leaves the pending set (never revised).
    - ``revise`` -> the verifier's ``suggested_revision`` is applied in-place
      (no separate rewrite call) and the query stays pending for re-verification,
      UNLESS it has no usable suggestion or has already been revised
      ``max_rewrites`` times — then it is discarded.

    Returns the ``RewriteRecord``s for revisions applied this round so the
    rewritten-queries export is unchanged.
    """
    rewrites: List[RewriteRecord] = []
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
        trace = traces[qid]
        suggestion = (result.suggested_revision or "").strip()
        if result.decision == "pass":
            trace.final_status = "accepted"
            trace.final_query_text = trace.current_query_text
            pending.discard(qid)
        elif (
            result.decision == "revise"
            and suggestion
            and trace.rewrite_count < max_rewrites
        ):
            # Apply the verifier's own suggested revision; re-verify next round.
            old_q = current_queries[qid]
            rewrites.append(
                RewriteRecord(
                    video_id=old_q.video_id,
                    query_id=qid,
                    round_index=round_index,
                    original_query_text=trace.current_query_text,
                    rewritten_query_text=suggestion,
                    query_type=old_q.query_type,
                    rewrite_reason=result.failure_reason,
                )
            )
            trace.rewrite_count += 1
            trace.current_query_text = suggestion
            current_queries[qid] = EventGroundedQuery(
                video_id=old_q.video_id,
                query_id=old_q.query_id,
                query_type=old_q.query_type,
                query_text=suggestion,
                grounding_event_description=old_q.grounding_event_description,
                approximate_grounding_time=old_q.approximate_grounding_time,
                target_person_or_group=old_q.target_person_or_group,
                expected_evidence=list(old_q.expected_evidence),
                why_grounded=old_q.why_grounded,
                time_range=list(old_q.time_range) if old_q.time_range else None,
                segment_ids=list(old_q.segment_ids),
                grounding_evidence=old_q.grounding_evidence,
                source_caption_ids=list(old_q.source_caption_ids),
                gen_time_range=list(old_q.gen_time_range),
                gen_segment_ids=list(old_q.gen_segment_ids),
            )
            # stays pending for re-verification
        else:
            # fail, or a revise we can't/shouldn't act on -> end here.
            trace.final_status = "discarded"
            trace.final_query_text = trace.current_query_text
            pending.discard(qid)
    return rewrites


def _query_uris(q: EventGroundedQuery, segment_uris: Dict[str, str]) -> List[str]:
    """The clip URIs for the segments a query is grounded on (in order)."""
    return [segment_uris[sid] for sid in q.segment_ids if sid in segment_uris]


def _chunks(items: list, size: int) -> List[list]:
    return [items[i : i + size] for i in range(0, len(items), max(1, size))]


def _verify_per_query(
    video_id: str,
    queries: List[EventGroundedQuery],
    segment_uris: Dict[str, str],
    round_index: int,
    client: BaseLLMClient,
    prompts_dir: Optional[Path],
    verify_parallel: int = 1,
) -> VerificationBatchOutput:
    """Verify queries, each watching only its own segment clip(s).

    Queries are grouped into batches of ``verify_parallel`` and each batch is run
    in one ``verify_queries_many`` call (truly batched on the Qwen3-Omni engine;
    sequential otherwise). A whole-batch failure falls back to one-at-a-time
    verification so a single bad query never drops the rest. Any query that gets
    NO result (call failed, or the model omitted/garbled it) is synthesized as a
    hard FAIL — a malformed verification output is treated as a failure, never a
    silent pass.
    """
    results: List[VerificationResult] = []
    for group in _chunks(queries, max(1, verify_parallel)):
        try:
            out = verify_queries_many(
                video_id, group, [_query_uris(q, segment_uris) for q in group],
                round_index, client, prompts_dir,
            )
            results.extend(out.results)
        except Exception as e:  # batch failed -> retry this group one-by-one
            print(f"    [verify batch fallback] {len(group)} query(ies): {e}")
            for q in group:
                try:
                    out = verify_queries(
                        video_id, _query_uris(q, segment_uris), [q],
                        round_index, client, prompts_dir,
                    )
                    results.extend(out.results)
                except Exception as e2:  # one bad query never aborts the video
                    print(f"    [verify fail-on-error] {q.query_id}: {e2}")

    # Format error / missing result -> hard FAIL (default to failure, not pass).
    seen = {r.query_id for r in results}
    for q in queries:
        if q.query_id not in seen:
            results.append(
                VerificationResult(
                    video_id=video_id,
                    query_id=q.query_id,
                    round_index=round_index,
                    decision="fail",
                    emotion_relevance_pass=False,
                    answerability_pass=False,
                    query_quality_pass=False,
                    failure_reason="verification output missing or invalid format",
                )
            )
    return VerificationBatchOutput(
        video_id=video_id, round_index=round_index, results=results
    )


def run_query_pipeline(
    video_id: str,
    gen_output: GenerationOutput,
    client: BaseLLMClient,
    segment_uris: Dict[str, str],
    max_rewrites: int = 3,
    max_accepted: int = 8,
    prompts_dir: Optional[Path] = None,
    verify_parallel: int = 1,
    verify_variant: str = "p7_rolecot",
) -> Tuple[
    Dict[str, QueryTrace],
    List[VerificationBatchOutput],
    List[RewriteBatchOutput],
]:
    """Run the verify(+revise) loop for one video's generated queries.

    ``segment_uris`` maps ``segment_id`` to the Files API URI of that segment's
    clip. Each query is verified against only the clips of its own ``segment_ids``,
    not the whole video. Verification defaults to the per-dimension architecture
    (``verify_variant``, default ``p7_rolecot`` — role + CoT; see
    ``prompts/perdim/``): relevance/query_quality are judged from the query text
    alone, answerability watches the clip(s), each in its own call.
    ``verify_parallel`` queries are grouped into one batched call per dimension
    (truly batched on the Qwen3-Omni engine). This path never produces a
    ``suggested_revision``, so a ``revise`` verdict is discarded rather than
    rewritten — there is no rewrite round in the default configuration, but
    ``max_rewrites`` rounds still run (each re-verifying whatever the previous
    round left pending) in case a caller passes a combined-call variant.
    """
    traces: Dict[str, QueryTrace] = {}
    current_queries: Dict[str, EventGroundedQuery] = {}
    for q in gen_output.queries:
        traces[q.query_id] = _make_trace(q)
        current_queries[q.query_id] = q

    pending: set = set(current_queries.keys())
    all_ver: List[VerificationBatchOutput] = []
    all_rw: List[RewriteBatchOutput] = []

    # One verify round per iteration. Round 1 is the initial verify; with the
    # default per-dimension variant there are no suggested_revisions to apply,
    # so pending empties out after round 1 (revise == discard, see module
    # docstring) — the loop still supports extra rounds for a combined-call
    # variant wired in by a caller.
    for round_index in range(1, max_rewrites + 2):
        if not pending:
            break
        round_queries = [current_queries[qid] for qid in sorted(pending)]
        ver_output = verify_queries_per_dimension(
            video_id, round_queries,
            [_query_uris(q, segment_uris) for q in round_queries],
            round_index, client, prompts_dir,
            variant=verify_variant, verify_parallel=verify_parallel,
        )
        all_ver.append(ver_output)
        rewrites = _apply_verification_results(
            ver_output.results, traces, pending, current_queries,
            round_index, max_rewrites,
        )
        if rewrites:
            all_rw.append(
                RewriteBatchOutput(
                    video_id=video_id, round_index=round_index, rewrites=rewrites
                )
            )

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
