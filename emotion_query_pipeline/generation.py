"""Query generation from observation captions + emotion events (text-only, no video).

The model never sees the video here — it reads the observation captions and the
emotion events (the Gemini emotion-event stage already did all emotion judgment)
and selects the moments worth turning into queries.

- Grounding handle is a **time range** ``[start, end]`` in seconds, not a
  ``segment_id``. Captions/events are shown with their time range, the model
  grounds each query on a time range, and we resolve it back to the overlapping
  ``segment_ids`` internally (each segment has exactly one caption / one clip).
  A query whose range is invalid or overlaps no segment is dropped.
- There is NO transcript anymore (WhisperX removed); no quotable-speech queries.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .io_utils import load_prompt_template
from .llm_client import BaseLLMClient
from .models import (
    EmotionCaption,
    EmotionEvent,
    EventGroundedQuery,
    GenerationOutput,
    OmniCaption,
    Segment,
)

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def _segment_time_map(segments: List[Segment]) -> Dict[str, Tuple[float, float]]:
    return {s.segment_id: (s.start_time, s.end_time) for s in segments}


# --- caption-type-agnostic accessors (works for OmniCaption + EmotionCaption) -
def _caption_segment_ids(caption) -> List[str]:
    if isinstance(caption, OmniCaption):
        return [caption.segment_id]
    return list(caption.segment_ids)


def _caption_id(caption, video_id: str) -> str:
    if isinstance(caption, OmniCaption):
        return f"{video_id}_{caption.segment_id}"
    return caption.caption_id


def _caption_time_range(
    caption, seg_time: Dict[str, Tuple[float, float]]
) -> Optional[List[float]]:
    """The [start, end] span a caption is grounded on (rounded)."""
    if isinstance(caption, OmniCaption):
        tr = caption.time_range
        if tr and len(tr) == 2:
            return [round(tr[0], 2), round(tr[1], 2)]
        span = seg_time.get(caption.segment_id)
        return [round(span[0], 2), round(span[1], 2)] if span else None
    spans = [seg_time[sid] for sid in caption.segment_ids if sid in seg_time]
    if not spans:
        return None
    return [round(min(s for s, _ in spans), 2), round(max(e for _, e in spans), 2)]


def _synthesize_visual_prose(caption) -> str:
    """Fallback visual prose for a caption with no ``visual_description``.

    Used only when ``visual_description`` is empty: a legacy structured
    ``OmniCaption`` (old cached run, ``visual_objective``/``visual_expression``/
    ``temporal_description`` populated instead) or a legacy flat
    ``EmotionCaption``. Keeps old cached runs serializing sensibly rather than
    emitting an empty ``caption`` string.
    """
    if isinstance(caption, OmniCaption):
        parts: List[str] = []
        vo = caption.visual_objective
        for p in vo.people:
            desc = " ".join(x for x in (p.person, p.action) if x).strip()
            if desc:
                parts.append(desc)
        if vo.scene and (vo.scene.location or vo.scene.setting):
            scene = " ".join(x for x in (vo.scene.location, vo.scene.setting) if x)
            if scene:
                parts.append(f"Scene: {scene}")
        for ve in caption.visual_expression:
            cues = [str(c) for c in list(ve.facial_cues) + list(ve.body_cues) if c]
            if cues:
                who = f"{ve.person}: " if ve.person else ""
                parts.append(f"{who}{', '.join(cues)}")
        legacy = getattr(caption, "temporal_description", "") or ""
        if legacy.strip():
            parts.append(legacy.strip())
        return " ".join(p for p in parts if p).strip()
    # Legacy flat EmotionCaption.
    parts = []
    person_action = " ".join(x for x in (caption.person, caption.action) if x).strip()
    if person_action:
        parts.append(person_action)
    if caption.observable_evidence:
        parts.append(", ".join(caption.observable_evidence))
    return " ".join(p for p in parts if p).strip()


def _caption_payload_entry(caption, tr: List[float]) -> dict:
    """One OBSERVATION caption as the model sees it — a unified schema for both types.

    The model-facing content is a single unstructured ``caption`` string:
    ``"Visual: <visual prose>\\nAudio: <audio prose>"`` (the ``Audio:`` line is
    omitted when there is no audio evidence). ``OmniCaption.visual_description``
    is the primary source for the visual prose; a legacy structured/flat caption
    with no ``visual_description`` falls back to ``_synthesize_visual_prose`` so
    old cached runs still serialize sensibly. NO emotion is included — emotion
    lives in the separate emotion-events payload.
    """
    if isinstance(caption, OmniCaption):
        visual = (caption.visual_description or "").strip()
        audio = (caption.audio_description or "").strip()
    else:
        visual = ""
        audio = (caption.sound or "").strip()
    if not visual:
        visual = _synthesize_visual_prose(caption)
    lines = []
    if visual:
        lines.append(f"Visual: {visual}")
    if audio:
        lines.append(f"Audio: {audio}")
    return {
        "time_range": tr,
        "caption": "\n".join(lines),
        "confidence": caption.confidence,
        "evidence_strength": caption.evidence_strength,
    }


def _captions_payload(
    captions: list, seg_time: Dict[str, Tuple[float, float]]
) -> list:
    """Structured, segment-level OBSERVATION evidence (no emotion).

    Accepts ``OmniCaption`` (rich, fed directly) or the legacy flat
    ``EmotionCaption``; both render to one unified observation schema. Shared by
    the emotion-event stage and the query stage.
    """
    payload = []
    for c in captions:
        tr = _caption_time_range(c, seg_time)
        if tr is None:
            continue  # no resolvable time range -> not groundable, skip
        payload.append(_caption_payload_entry(c, tr))
    return payload


def _events_payload(events: Optional[List[EmotionEvent]]) -> list:
    """Emotion events as the query model sees them (the emotion signal)."""
    out = []
    for e in events or []:
        out.append(
            {
                "event_id": e.event_id,
                "emotion_label": e.emotion_label,
                "event_description": e.event_description,
                "time_range": e.time_range,
                "target_person_or_group": e.target_person_or_group,
                "visual_evidence": list(e.visual_evidence),
                "audio_evidence": list(e.audio_evidence),
                "confidence": e.confidence,
                "evidence_strength": e.evidence_strength,
            }
        )
    return out


def build_generation_prompt(
    video_id: str,
    captions: List[EmotionCaption],
    events: Optional[List[EmotionEvent]],
    segments: List[Segment],
    prompts_dir: Optional[Path] = None,
) -> str:
    template = load_prompt_template(
        prompts_dir or _PROMPTS_DIR, "generation_prompt.txt"
    )
    seg_time = _segment_time_map(segments)
    prompt = template
    prompt = prompt.replace("{video_id}", video_id)
    prompt = prompt.replace(
        "{captions_json}",
        json.dumps(_captions_payload(captions, seg_time), indent=2, ensure_ascii=False),
    )
    prompt = prompt.replace(
        "{events_json}",
        json.dumps(_events_payload(events), indent=2, ensure_ascii=False),
    )
    return prompt


def generate_queries(
    video_id: str,
    captions: List[EmotionCaption],
    events: Optional[List[EmotionEvent]],
    client: BaseLLMClient,
    segments: List[Segment],
    prompts_dir: Optional[Path] = None,
) -> GenerationOutput:
    """Generate queries from observation captions + emotion events.

    Returns a validated GenerationOutput. With no emotion events there is nothing
    to ground emotion queries on, so an empty list is returned.
    """
    if not captions or not events:
        return GenerationOutput(video_id=video_id, queries=[])

    prompt = build_generation_prompt(
        video_id, captions, events, segments, prompts_dir
    )
    raw = client.generate_json(prompt, "GenerationOutput", video_uri=None)
    raw.setdefault("video_id", video_id)
    _valid_query_types = {"explicit_event", "emotion_state", "evidence_cue"}
    cleaned_queries = []
    dropped = 0
    for i, q in enumerate(raw.get("queries") or [], 1):
        q.setdefault("video_id", video_id)
        # The model occasionally omits query_id; fill a deterministic one so a
        # single missing id doesn't fail validation for the whole video.
        if not q.get("query_id"):
            q["query_id"] = f"{video_id}_q{i:02d}"
        # It also occasionally emits an out-of-enum query_type (e.g. an emotion
        # label like 'happy' instead of one of explicit_event/emotion_state/
        # evidence_cue). Drop just that query rather than letting one bad value
        # fail GenerationOutput validation for the ENTIRE video.
        if q.get("query_type") not in _valid_query_types:
            dropped += 1
            continue
        cleaned_queries.append(q)
    if dropped:
        print(f"  [generation] dropped {dropped} query(ies) with invalid "
              f"query_type for {video_id}")
    raw["queries"] = cleaned_queries

    output = GenerationOutput.model_validate(raw)
    return _resolve_time_ranges(output, segments, captions)


def _resolve_time_ranges(
    output: GenerationOutput,
    segments: List[Segment],
    captions: Optional[List[EmotionCaption]] = None,
) -> GenerationOutput:
    """Validate each query's time_range and resolve internal grounding handles.

    Fills (a) ``segment_ids`` = the segments overlapping the query's time range
    (for verification clip lookup) and (b) ``source_caption_ids`` = the caption
    ids on those segments (provenance/debug). Drops queries whose ``time_range``
    is missing/malformed, lies outside the video, or overlaps no real segment.
    """
    if not segments:
        output.queries = []
        return output
    video_end = max(s.end_time for s in segments)
    # segment_id -> caption_id (each segment carries exactly one caption).
    seg_to_caption: Dict[str, str] = {}
    for c in captions or []:
        cid = _caption_id(c, output.video_id)
        for sid in _caption_segment_ids(c):
            seg_to_caption.setdefault(sid, cid)
    kept: List[EventGroundedQuery] = []
    for q in output.queries:
        tr = q.time_range
        if not tr or len(tr) != 2:
            continue
        try:
            qs, qe = float(tr[0]), float(tr[1])
        except (TypeError, ValueError):
            continue
        # Clamp to the video and require a positive-length, in-bounds range.
        qs = max(0.0, qs)
        qe = min(video_end, qe)
        if qe <= qs:
            continue
        covering = sorted(
            (s for s in segments if s.start_time < qe and s.end_time > qs),
            key=lambda s: s.index,
        )
        if not covering:
            continue
        q.time_range = [round(qs, 2), round(qe, 2)]
        q.segment_ids = [s.segment_id for s in covering]
        q.source_caption_ids = [
            seg_to_caption[s.segment_id]
            for s in covering
            if s.segment_id in seg_to_caption
        ]
        kept.append(q)
    output.queries = kept
    return output
