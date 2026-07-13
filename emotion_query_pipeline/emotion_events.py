"""Gemini emotion-event stage — the ONLY place emotion is judged.

Text-only (no video): reads the OBSERVATION captions of one video and produces
``EmotionEvent``s, each labelled with one of the eight emotion-relevant classes.
Moments without clear emotion-relevant evidence simply yield no event. Each
event's ``time_range`` is resolved back to overlapping ``segment_ids`` for clip
lookup / provenance. The resulting events feed the query stage.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from .generation import _captions_payload, _segment_time_map
from .io_utils import load_prompt_template
from .llm_client import BaseLLMClient
from .models import EmotionEvent, EmotionEventOutput, Segment

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

# Runaway-generation guard: emotion events GROUP captions, so a healthy run yields
# at most ~one event per caption (events <= captions). Gemini occasionally goes into
# a degenerate sampling loop on long/verbose caption inputs and emits FAR more events
# than captions (observed: 57 captions -> 180 events, a 212KB response, in one draw;
# a clean re-draw of the same input gave 17). We treat any draw above this ceiling as
# runaway and re-sample (a fresh draw is usually fine), keeping the least-inflated
# attempt. Floor keeps short videos (a handful of captions) from tripping it.
_DEGENERATE_EVENT_MULTIPLE = 2
_DEGENERATE_EVENT_FLOOR = 10
_MAX_EVENT_ATTEMPTS = 3


def _parse_and_validate_events(raw: dict, video_id: str) -> List[EmotionEvent]:
    """Validate the raw events ONE AT A TIME, dropping invalid ones.

    A single bad event (e.g. an emotion_label outside the eight allowed classes)
    is dropped rather than failing the whole video.
    """
    events: List[EmotionEvent] = []
    for i, e in enumerate(raw.get("events") or [], 1):
        if not isinstance(e, dict):
            continue
        e.setdefault("video_id", video_id)
        if not e.get("event_id"):
            e["event_id"] = f"{video_id}_e{i:02d}"
        try:
            events.append(EmotionEvent.model_validate(e))
        except Exception as ex:
            print(f"[emotion_event] dropped invalid event "
                  f"(label={e.get('emotion_label')!r}): {ex}")
    return events


_CONFIDENCE_RANK = {"high": 3, "medium": 2, "low": 1}
_EVIDENCE_RANK = {"clear": 3, "ambiguous": 2, "weak": 1}


def _normalize_target(target: str) -> str:
    return (target or "").strip().lower()


def _events_should_merge(a: EmotionEvent, b: EmotionEvent, max_gap: float) -> bool:
    """Whether ``b`` (starting no earlier than ``a``) continues the same event.

    Same ``emotion_label``, same (normalized) ``target_person_or_group``, and
    contiguous/overlapping time ranges — the gap between ``a``'s end and ``b``'s
    start is at most ``max_gap``. Callers pass half a segment-length, so touching
    segments (gap ~0) merge but a whole missing segment in between (gap
    ~seg_len) does not.
    """
    if a.emotion_label != b.emotion_label:
        return False
    if _normalize_target(a.target_person_or_group) != _normalize_target(b.target_person_or_group):
        return False
    a_start, a_end = a.time_range
    b_start, b_end = b.time_range
    gap = b_start - a_end
    return gap <= max_gap


def _dedup_evidence(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _merge_two_events(a: EmotionEvent, b: EmotionEvent) -> EmotionEvent:
    """Merge event ``b`` into event ``a`` (``a`` starts no later than ``b``)."""
    a_start, a_end = a.time_range
    b_start, b_end = b.time_range
    merged_range = [min(a_start, b_start), max(a_end, b_end)]
    if a.event_description == b.event_description or not b.event_description:
        merged_desc = a.event_description
    elif not a.event_description:
        merged_desc = b.event_description
    else:
        merged_desc = f"{a.event_description} {b.event_description}".strip()
    conf = a.confidence if _CONFIDENCE_RANK[a.confidence] >= _CONFIDENCE_RANK[b.confidence] else b.confidence
    ev_strength = (
        a.evidence_strength
        if _EVIDENCE_RANK[a.evidence_strength] >= _EVIDENCE_RANK[b.evidence_strength]
        else b.evidence_strength
    )
    return a.model_copy(
        update={
            "time_range": merged_range,
            "event_description": merged_desc,
            "visual_evidence": _dedup_evidence(list(a.visual_evidence) + list(b.visual_evidence)),
            "audio_evidence": _dedup_evidence(list(a.audio_evidence) + list(b.audio_evidence)),
            "segment_ids": _dedup_evidence(list(a.segment_ids) + list(b.segment_ids)),
            "confidence": conf,
            "evidence_strength": ev_strength,
        }
    )


def merge_contiguous_events(
    events: List[EmotionEvent], max_gap: float = 5.0
) -> List[EmotionEvent]:
    """Deterministic backstop: merge adjacent same-emotion event continuations.

    Gemini is asked to merge consecutive same-emotion segments into one event
    (see the prompt rule), but almost never does on its own — this runs AFTER
    validation/the runaway guard and composes cleanly with it: it only ever
    reduces the event count, never invents or drops emotion-relevant content
    (evidence/segment_ids are unioned, not discarded). Events without a valid
    2-element ``time_range`` are left untouched (can't be ordered/merged) and
    passed through as-is.
    """
    if not events:
        return []
    with_range = [e for e in events if e.time_range and len(e.time_range) == 2]
    without_range = [e for e in events if not (e.time_range and len(e.time_range) == 2)]
    with_range.sort(key=lambda e: e.time_range[0])

    merged: List[EmotionEvent] = []
    for e in with_range:
        if merged and _events_should_merge(merged[-1], e, max_gap):
            merged[-1] = _merge_two_events(merged[-1], e)
        else:
            merged.append(e)
    return merged + without_range


def build_emotion_event_prompt(
    video_id: str,
    captions: list,
    segments: List[Segment],
    prompts_dir: Optional[Path] = None,
) -> str:
    template = load_prompt_template(
        prompts_dir or _PROMPTS_DIR, "emotion_event_prompt.txt"
    )
    seg_time = _segment_time_map(segments)
    prompt = template
    prompt = prompt.replace("{video_id}", video_id)
    prompt = prompt.replace(
        "{captions_json}",
        json.dumps(_captions_payload(captions, seg_time), indent=2, ensure_ascii=False),
    )
    return prompt


def generate_emotion_events(
    video_id: str,
    captions: list,
    client: BaseLLMClient,
    segments: List[Segment],
    prompts_dir: Optional[Path] = None,
) -> EmotionEventOutput:
    """Infer emotion events from observation captions. Returns validated output."""
    if not captions:
        return EmotionEventOutput(video_id=video_id, events=[])

    prompt = build_emotion_event_prompt(video_id, captions, segments, prompts_dir)
    ceiling = max(_DEGENERATE_EVENT_MULTIPLE * len(captions), _DEGENERATE_EVENT_FLOOR)

    # Re-sample if a draw is runaway (see the guard constants above). Keep the
    # least-inflated attempt across tries — a runaway draw has FAR more events than a
    # healthy one, so "fewest events" reliably picks the good draw when one appears.
    events: List[EmotionEvent] = []
    for attempt in range(1, _MAX_EVENT_ATTEMPTS + 1):
        raw = client.generate_json(prompt, "EmotionEventOutput", video_uri=None)
        attempt_events = _parse_and_validate_events(raw, video_id)
        if not events or len(attempt_events) < len(events):
            events = attempt_events
        if len(attempt_events) <= ceiling:
            break
        print(f"[emotion_event] {video_id}: {len(attempt_events)} events > ceiling "
              f"{ceiling} ({len(captions)} captions) — likely runaway generation, "
              f"re-sampling (attempt {attempt}/{_MAX_EVENT_ATTEMPTS})")

    if len(events) > ceiling:
        # Every attempt was runaway — keep the least-bad one but truncate to the
        # ceiling so the downstream query/verify stages aren't flooded.
        print(f"[emotion_event] {video_id}: still {len(events)} events after "
              f"{_MAX_EVENT_ATTEMPTS} attempts; truncating to {ceiling}")
        events = events[:ceiling]

    seg_len = (segments[0].end_time - segments[0].start_time) if segments else 5.0
    # Half a segment: merges only touching/overlapping continuations (gap ~0),
    # never bridges a whole segment that produced no event (gap ~seg_len), which
    # would falsely stretch the emotion's time_range across a neutral stretch.
    events = merge_contiguous_events(events, max_gap=seg_len * 0.5)

    output = EmotionEventOutput(video_id=video_id, events=events)
    return _resolve_event_segments(output, segments)


def _resolve_event_segments(
    output: EmotionEventOutput, segments: List[Segment]
) -> EmotionEventOutput:
    """Fill each event's ``segment_ids`` from its ``time_range`` (overlap).

    Events are kept even if the range is missing/unresolvable (emotion judgment
    is still useful context for query generation); only ``segment_ids`` is left
    empty in that case.
    """
    if not segments:
        return output
    video_end = max(s.end_time for s in segments)
    for e in output.events:
        tr = e.time_range
        if not tr or len(tr) != 2:
            continue
        try:
            es, ee = float(tr[0]), float(tr[1])
        except (TypeError, ValueError):
            continue
        es = max(0.0, es)
        ee = min(video_end, ee)
        if ee <= es:
            continue
        covering = sorted(
            (s for s in segments if s.start_time < ee and s.end_time > es),
            key=lambda s: s.index,
        )
        e.time_range = [round(es, 2), round(ee, 2)]
        e.segment_ids = [s.segment_id for s in covering]
    return output
