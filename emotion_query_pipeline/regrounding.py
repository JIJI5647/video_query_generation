"""Re-grounding stage — a Gemini call re-selects each query's grounding.

Runs AFTER generation and BEFORE verification. Generation's own time_range/
segment_ids grounding is unreliable (the model writes ``time_range`` from
memory while composing the query, not from a dedicated grounding pass); this
stage feeds each query's TEXT plus the video's OBSERVATION captions (no video,
text-only, same as generation/emotion_events) back to Gemini and lets it
re-pick the grounding segment(s) with the query already fixed. The result
becomes the FINAL grounding that verification checks against; the original
generation-stage grounding is preserved in ``gen_time_range``/``gen_segment_ids``
for offline comparison (see ``EventGroundedQuery`` in ``models.py``).

ONE call per video, batched over all of that video's queries (never one call
per query) — see ``build_regrounding_prompt``. Two scopes, selected by the
caller:

  * "full"   — every query's candidate list is ALL of the video's segments;
               Gemini may pick any 1 or several CONTIGUOUS segments.
  * "window" — every query's candidate list is restricted to the segments
               within +/- ``window`` of that query's OWN current grounding.
               Different queries have different windows; rather than one call
               per query (which would violate the batching rule), each query
               in the single-call payload carries its own ``candidate_segments``
               list, pre-filtered to its window. Gemini is still one call for
               the whole video in both scopes.

Robustness: a missing/invalid/empty/out-of-window/non-contiguous selection for
a query FALLS BACK to that query's original (generation-stage) grounding —
never dropped, never crashes the video. The number of fallbacks is returned so
callers can log/aggregate it.

Pure / import-safe / unit-testable with a fake ``BaseLLMClient``, exactly like
``generation.py`` and ``emotion_events.py``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .generation import (
    _caption_id,
    _caption_payload_entry,
    _caption_segment_ids,
    _caption_time_range,
    _segment_time_map,
)
from .io_utils import load_prompt_template
from .llm_client import BaseLLMClient
from .models import EventGroundedQuery, RegroundingOutput, Segment

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def _captions_by_segment(captions: list, seg_time: Dict[str, Tuple[float, float]]) -> Dict[str, dict]:
    """One caption payload entry per ``segment_id`` (each segment has one caption).

    Reuses the same unified "Visual: ...\\nAudio: ..." rendering generation.py
    uses, plus the ``segment_id`` (generation's payload omits it — the query
    model never needs it, but re-grounding returns segment_ids, so it must be
    shown here).
    """
    by_seg: Dict[str, dict] = {}
    for c in captions:
        tr = _caption_time_range(c, seg_time)
        if tr is None:
            continue
        entry = _caption_payload_entry(c, tr)
        for sid in _caption_segment_ids(c):
            span = seg_time.get(sid)
            if span is None:
                continue
            e = dict(entry)
            e["segment_id"] = sid
            e["time_range"] = [round(span[0], 2), round(span[1], 2)]
            by_seg[sid] = e
    return by_seg


def _ordered_segment_ids(segments: List[Segment]) -> List[str]:
    return [s.segment_id for s in sorted(segments, key=lambda s: s.index)]


def _window_segment_ids(
    orig_segment_ids: List[str], ordered_ids: List[str], window: int
) -> List[str]:
    """The +/- ``window`` segment ids around a query's current grounding.

    Falls back to the full ordered list if the query has no resolvable current
    segment_ids (e.g. regrounding a query that generation failed to ground) so
    it still gets SOME candidates rather than an empty window.
    """
    idx_of = {sid: i for i, sid in enumerate(ordered_ids)}
    orig_idxs = [idx_of[sid] for sid in orig_segment_ids if sid in idx_of]
    if not orig_idxs:
        return list(ordered_ids)
    lo = max(0, min(orig_idxs) - window)
    hi = min(len(ordered_ids) - 1, max(orig_idxs) + window)
    return ordered_ids[lo : hi + 1]


def build_regrounding_prompt(
    video_id: str,
    queries: List[EventGroundedQuery],
    captions: list,
    segments: List[Segment],
    scope: str = "full",
    window: int = 2,
    prompts_dir: Optional[Path] = None,
) -> str:
    """Build the single per-video re-grounding prompt.

    Each query in the payload carries its OWN ``candidate_segments`` list —
    every segment for "full" scope, the +/- ``window`` segments around the
    query's current grounding for "window" scope — so one call covers every
    query regardless of scope.
    """
    template = load_prompt_template(prompts_dir or _PROMPTS_DIR, "regrounding_prompt.txt")
    seg_time = _segment_time_map(segments)
    by_seg = _captions_by_segment(captions, seg_time)
    ordered_ids = _ordered_segment_ids(segments)

    queries_payload = []
    for q in queries:
        if scope == "window":
            cand_ids = _window_segment_ids(q.segment_ids, ordered_ids, window)
        else:
            cand_ids = ordered_ids
        queries_payload.append(
            {
                "query_id": q.query_id,
                "query_text": q.query_text,
                "candidate_segments": [by_seg[sid] for sid in cand_ids if sid in by_seg],
            }
        )

    prompt = template
    prompt = prompt.replace("{video_id}", video_id)
    prompt = prompt.replace(
        "{queries_json}",
        json.dumps(queries_payload, indent=2, ensure_ascii=False),
    )
    return prompt


def _valid_contiguous_selection(
    picked: List[str],
    allowed: Optional[set],
    seg_index: Dict[str, int],
) -> bool:
    """Non-empty, every id known, every id inside ``allowed`` (if given), contiguous."""
    if not picked:
        return False
    if any(sid not in seg_index for sid in picked):
        return False
    if allowed is not None and any(sid not in allowed for sid in picked):
        return False
    idxs = sorted(seg_index[sid] for sid in picked)
    if len(set(idxs)) != len(idxs):
        return False  # duplicate segment id
    return idxs[-1] - idxs[0] + 1 == len(idxs)  # consecutive run, no gaps


def reground_queries(
    video_id: str,
    queries: List[EventGroundedQuery],
    captions: list,
    segments: List[Segment],
    client: BaseLLMClient,
    scope: str = "full",
    window: int = 2,
    prompts_dir: Optional[Path] = None,
) -> Tuple[List[EventGroundedQuery], Dict[str, int]]:
    """Re-select each query's grounding via ONE Gemini call for the whole video.

    Returns ``(updated_queries, stats)`` where ``stats`` is
    ``{"total": N, "changed": C, "fallback": F}`` — ``changed`` counts queries
    whose final segment_ids differ from their original grounding (a real
    re-selection, not a fallback); ``fallback`` counts queries where Gemini's
    selection was missing/invalid and the original grounding was kept.
    Every input query is preserved (never dropped), and ``gen_time_range`` /
    ``gen_segment_ids`` are set on every output query to its ORIGINAL grounding
    before any overwrite, regardless of whether the re-selection succeeded.
    """
    total = len(queries)
    if not total or not segments:
        return list(queries), {"total": total, "changed": 0, "fallback": 0}

    seg_time = _segment_time_map(segments)
    seg_index = {s.segment_id: s.index for s in segments}
    ordered_ids = _ordered_segment_ids(segments)

    prompt = build_regrounding_prompt(
        video_id, queries, captions, segments, scope=scope, window=window,
        prompts_dir=prompts_dir,
    )
    try:
        raw = client.generate_json(prompt, "RegroundingOutput", video_uri=None)
        raw.setdefault("video_id", video_id)
        parsed = RegroundingOutput.model_validate(raw)
        chosen: Dict[str, List[str]] = {
            g.query_id: list(g.segment_ids) for g in parsed.groundings
        }
    except Exception as e:  # a whole-video call failure -> fall back for everyone
        print(f"  [reground] call failed for {video_id}: {e} — keeping original "
              f"grounding for all {total} quer(y/ies)")
        chosen = {}

    updated: List[EventGroundedQuery] = []
    changed = 0
    fallback = 0
    for q in queries:
        new_q = q.model_copy(deep=True)
        new_q.gen_time_range = list(q.time_range) if q.time_range else []
        new_q.gen_segment_ids = list(q.segment_ids)

        picked = chosen.get(q.query_id) or []
        allowed = (
            set(_window_segment_ids(q.segment_ids, ordered_ids, window))
            if scope == "window" else None
        )
        if not _valid_contiguous_selection(picked, allowed, seg_index):
            fallback += 1
            updated.append(new_q)  # time_range/segment_ids left as the original
            continue

        picked_sorted = sorted(picked, key=lambda sid: seg_index[sid])
        spans = [seg_time[sid] for sid in picked_sorted]
        new_q.time_range = [round(min(s for s, _ in spans), 2), round(max(e for _, e in spans), 2)]
        new_q.segment_ids = picked_sorted
        new_q.source_caption_ids = [
            _caption_id(c, video_id)
            for c in captions
            if set(_caption_segment_ids(c)) & set(picked_sorted)
        ]
        if picked_sorted != new_q.gen_segment_ids:
            changed += 1
        updated.append(new_q)

    stats = {"total": total, "changed": changed, "fallback": fallback}
    if fallback:
        print(f"  [reground] {fallback}/{total} quer(y/ies) fell back to original "
              f"grounding for {video_id}")
    return updated, stats
