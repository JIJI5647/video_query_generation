"""Compute aggregate pipeline statistics (caption-aware for v2)."""
from __future__ import annotations

from collections import Counter
from typing import Dict, List

from .models import (
    EmotionEventOutput,
    GenerationOutput,
    PipelineStats,
    QueryTrace,
    Segment,
    VerificationBatchOutput,
)


def compute_stats(
    video_traces: Dict[str, Dict[str, QueryTrace]],
    gen_outputs: Dict[str, GenerationOutput],
    ver_outputs: Dict[str, List[VerificationBatchOutput]],
    segments: Dict[str, List[Segment]],
    raw_captions: Dict[str, list],
    emotion_events: Dict[str, EmotionEventOutput],
    validation_warnings: List[str],
    regrounding: Dict[str, Dict[str, int]] | None = None,
) -> PipelineStats:
    total_videos = len(video_traces)
    total_initial = sum(
        len(gen_outputs[vid].queries)
        for vid in video_traces
        if vid in gen_outputs
    )

    all_traces: List[QueryTrace] = [
        t for traces in video_traces.values() for t in traces.values()
    ]
    accepted = [t for t in all_traces if t.final_status == "accepted"]
    discarded = [t for t in all_traces if t.final_status == "discarded"]

    total_accepted = len(accepted)
    total_discarded = len(discarded)
    avg_accepted = total_accepted / total_videos if total_videos else 0.0

    # Caption-stage tallies (observation captions — no emotion on captions).
    total_segments = sum(len(v) for v in segments.values())
    total_raw = sum(len(v) for v in raw_captions.values())
    # Emotion distribution now comes from the emotion-event stage (the 8 labels).
    total_events = sum(len(out.events) for out in emotion_events.values())
    emotion_dist: Counter = Counter()
    for out in emotion_events.values():
        for e in out.events:
            emotion_dist[e.emotion_label] += 1

    # Query type distributions
    initial_types: Counter = Counter()
    for gen_out in gen_outputs.values():
        for q in gen_out.queries:
            initial_types[q.query_type] += 1
    final_accepted_types: Counter = Counter(t.query_type for t in accepted)

    # Rewrite count distribution
    rewrite_dist: Counter = Counter(t.rewrite_count for t in all_traces)

    # Pass rate after initial verification (round 1)
    passed_round1 = 0
    total_round1 = 0
    for ver_list in ver_outputs.values():
        if ver_list:
            r1 = ver_list[0]
            total_round1 += len(r1.results)
            passed_round1 += sum(1 for r in r1.results if r.decision == "pass")
    pass_rate_initial = passed_round1 / total_round1 if total_round1 else 0.0

    pass_rate_after_rewrites = total_accepted / total_initial if total_initial else 0.0

    diversity_warnings = [w for w in validation_warnings if "WARNING" in w]

    # Re-grounding stage (0/0 if disabled — regrounding stays {}).
    regrounding_changed = sum(v.get("changed", 0) for v in (regrounding or {}).values())
    regrounding_fallback = sum(v.get("fallback", 0) for v in (regrounding or {}).values())

    return PipelineStats(
        total_videos=total_videos,
        total_segments=total_segments,
        total_raw_captions=total_raw,
        total_emotion_events=total_events,
        total_initial_queries=total_initial,
        total_accepted_queries=total_accepted,
        total_discarded_queries=total_discarded,
        average_accepted_queries_per_video=round(avg_accepted, 4),
        emotion_distribution=dict(emotion_dist),
        query_type_distribution_initial=dict(initial_types),
        query_type_distribution_final_accepted=dict(final_accepted_types),
        rewrite_count_distribution={str(k): v for k, v in sorted(rewrite_dist.items())},
        pass_rate_after_initial_verification=round(pass_rate_initial, 4),
        pass_rate_after_rewrites=round(pass_rate_after_rewrites, 4),
        discarded_query_count=total_discarded,
        diversity_warnings=diversity_warnings,
        regrounding_changed_count=regrounding_changed,
        regrounding_fallback_count=regrounding_fallback,
    )
