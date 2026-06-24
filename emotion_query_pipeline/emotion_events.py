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
from .models import EmotionEventOutput, Segment

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


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
    raw = client.generate_json(prompt, "EmotionEventOutput", video_uri=None)
    raw.setdefault("video_id", video_id)
    for i, e in enumerate(raw.get("events") or [], 1):
        e.setdefault("video_id", video_id)
        if not e.get("event_id"):
            e["event_id"] = f"{video_id}_e{i:02d}"

    output = EmotionEventOutput.model_validate(raw)
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
