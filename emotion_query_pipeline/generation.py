"""Step 5: generate queries from all of a video's captions (text-only, no video).

The model never sees the video here — it reads every emotion caption extracted
from the video (plus the spoken-dialogue transcript, when available) and selects
the moments worth turning into queries.

v4 changes:
- Grounding handle is a **time range** ``[start, end]`` in seconds, not a
  ``segment_id``. Captions are shown to the model with their time range, and the
  model grounds each query on a time range. We validate every query's range and
  resolve it back to the overlapping ``segment_ids`` internally (each segment has
  exactly one caption / one clip), so the rest of the pipeline — which uploads
  and verifies per-segment clips — is unchanged. A query whose range is invalid
  or overlaps no segment is dropped.
- A WhisperX dialogue transcript is spliced into the prompt as extra context.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .io_utils import load_prompt_template
from .llm_client import BaseLLMClient
from .models import EmotionCaption, EventGroundedQuery, GenerationOutput, Segment

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def _segment_time_map(segments: List[Segment]) -> Dict[str, Tuple[float, float]]:
    return {s.segment_id: (s.start_time, s.end_time) for s in segments}


def _caption_time_range(
    caption: EmotionCaption, seg_time: Dict[str, Tuple[float, float]]
) -> Optional[List[float]]:
    """The [min start, max end] span of the segments a caption is grounded on."""
    spans = [seg_time[sid] for sid in caption.segment_ids if sid in seg_time]
    if not spans:
        return None
    return [round(min(s for s, _ in spans), 2), round(max(e for _, e in spans), 2)]


def _captions_payload(
    captions: List[EmotionCaption], seg_time: Dict[str, Tuple[float, float]]
) -> list:
    payload = []
    for c in captions:
        tr = _caption_time_range(c, seg_time)
        if tr is None:
            continue  # no resolvable time range -> not groundable, skip
        payload.append(
            {
                "time_range": tr,
                "person": c.person,
                "action": c.action,
                "sound": c.sound,
                "emotion": c.emotion,
                "confidence": c.confidence,
                "evidence_strength": c.evidence_strength,
                "observable_evidence": c.observable_evidence,
            }
        )
    return payload


def _transcript_payload(transcript: Optional[List[dict]]) -> list:
    if not transcript:
        return []
    out = []
    for line in transcript:
        text = (line.get("text") or "").strip()
        if not text:
            continue
        out.append(
            {
                "time_range": [
                    round(float(line.get("start", 0.0)), 2),
                    round(float(line.get("end", 0.0)), 2),
                ],
                "text": text,
            }
        )
    return out


def build_generation_prompt(
    video_id: str,
    captions: List[EmotionCaption],
    segments: List[Segment],
    transcript: Optional[List[dict]] = None,
    prompts_dir: Optional[Path] = None,
) -> str:
    template = load_prompt_template(
        prompts_dir or _PROMPTS_DIR, "generation_prompt.txt"
    )
    seg_time = _segment_time_map(segments)
    tx = _transcript_payload(transcript)
    transcript_json = (
        json.dumps(tx, indent=2, ensure_ascii=False)
        if tx
        else "(no spoken dialogue transcript available for this video)"
    )
    prompt = template
    prompt = prompt.replace("{video_id}", video_id)
    prompt = prompt.replace(
        "{captions_json}",
        json.dumps(_captions_payload(captions, seg_time), indent=2, ensure_ascii=False),
    )
    prompt = prompt.replace("{transcript_json}", transcript_json)
    return prompt


def generate_queries(
    video_id: str,
    captions: List[EmotionCaption],
    client: BaseLLMClient,
    segments: List[Segment],
    transcript: Optional[List[dict]] = None,
    prompts_dir: Optional[Path] = None,
) -> GenerationOutput:
    """Generate queries from all of a video's captions. Returns a validated GenerationOutput."""
    if not captions:
        return GenerationOutput(video_id=video_id, queries=[])

    prompt = build_generation_prompt(
        video_id, captions, segments, transcript, prompts_dir
    )
    raw = client.generate_json(prompt, "GenerationOutput", video_uri=None)
    raw.setdefault("video_id", video_id)
    for i, q in enumerate(raw.get("queries") or [], 1):
        q.setdefault("video_id", video_id)
        # The model occasionally omits query_id; fill a deterministic one so a
        # single missing id doesn't fail validation for the whole video.
        if not q.get("query_id"):
            q["query_id"] = f"{video_id}_q{i:02d}"

    output = GenerationOutput.model_validate(raw)
    return _resolve_time_ranges(output, segments)


def _resolve_time_ranges(
    output: GenerationOutput, segments: List[Segment]
) -> GenerationOutput:
    """Validate each query's time_range and resolve it to overlapping segment_ids.

    Drops queries whose ``time_range`` is missing/malformed, lies outside the
    video, or overlaps no real segment. Guarantees every surviving query has a
    valid ``time_range`` AND a non-empty internal ``segment_ids`` for clip lookup.
    """
    if not segments:
        output.queries = []
        return output
    video_end = max(s.end_time for s in segments)
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
        kept.append(q)
    output.queries = kept
    return output
