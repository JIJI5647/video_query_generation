"""Reference-free LLM/MLLM-as-judge scoring for caption and emotion-event quality.

Design (literature synthesis — VELA, CapArena, G-VEval, FLEUR, VDCScore), see
``prompts/judge/caption_quality_judge.txt`` / ``prompts/judge/emotion_event_judge.txt``
for the exact rubric wording:

- POINTWISE, one judge call per (clip, item), all dimensions returned together as
  structured JSON with a ``reason`` field BEFORE each ``score`` (rationale-first,
  CapArena/G-VEval).
- Caption dimensions: faithfulness, hallucination (kept separate + weighted
  heavily, CapArena), coverage, fluency (lowest priority), emotion_leakage_ok
  (endpoint-only 0/1, FLEUR-style, but checking OUR observational-caption
  constraint rather than emotion accuracy).
- Emotion-event dimensions: cue_sufficiency, cue_grounded, label_agreement. The
  judge is asked to independently name a best-fit label from the clip + cues
  ALONE (not shown for this sub-task which label was assigned); agreement is
  then computed HERE in code by string comparison, not self-reported by the
  model — matches this repo's existing pattern of deriving decisions in code
  from raw model fields (see ``verification.py:_decision_from_dimensions``)
  rather than trusting a self-scored verdict. Structural port of VDCScore's
  answer-then-check, made reference-free.
- Bias mitigations baked into the prompts: ignore caption length/verbosity
  (CapArena), reason-before-score (CapArena/G-VEval), each dimension judged on
  its own merits (VELA dimension isolation via separate reason fields in one
  call), no ground-truth reference — judge only against the clip.

Everything in this module is pure Python: sampling, prompt building, JSON
parsing/validation, aggregation. No ``google.genai`` import here — the CLI
(``judge_captions.py``) wires in ``GeminiLLMClient`` / ``GeminiUploader`` for the
actual calls, so this module is directly testable with a fake client.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .io_utils import load_prompt_template

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

CAPTION_JUDGE_PROMPT_FILE = "judge/caption_quality_judge.txt"
EVENT_JUDGE_PROMPT_FILE = "judge/emotion_event_judge.txt"

CAPTION_SCORE_DIMENSIONS: tuple[str, ...] = (
    "faithfulness", "hallucination", "coverage", "fluency", "emotion_leakage_ok",
)
EVENT_SCORE_DIMENSIONS: tuple[str, ...] = (
    "cue_sufficiency", "cue_grounded", "label_agreement",
)

# Combined caption score = weighted mean of the 1-5 dimensions (emotion_leakage_ok
# is 0/1 and reported separately, not blended into the 1-5 scale). hallucination is
# weighted the heaviest per CapArena's "keep it separate + weight it heavily".
CAPTION_COMBINED_WEIGHTS: Dict[str, float] = {
    "faithfulness": 0.30,
    "hallucination": 0.35,
    "coverage": 0.20,
    "fluency": 0.15,
}

EIGHT_EMOTION_LABELS: tuple[str, ...] = (
    "angry", "excited", "fear", "sad", "surprised", "frustrated", "happy",
    "disappointed",
)


# ---------------------------------------------------------------------------
# Field access — items may be dicts (loaded from JSONL) or objects/pydantic
# models (e.g. OmniCaption, EmotionEvent instances).
# ---------------------------------------------------------------------------
def _get(item: Any, key: str, default: Any = "") -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


# ---------------------------------------------------------------------------
# Sampling — deterministic round-robin spread across videos, no randomness.
# ---------------------------------------------------------------------------
def sample_segments(
    items: Sequence[Any],
    n: int,
    id_key: str = "segment_id",
    video_key: str = "video_id",
) -> List[Any]:
    """Deterministically sample up to ``n`` items, spread evenly across videos.

    Groups by ``video_key``, sorts each video's items by (``video_key``,
    ``id_key``) for reproducibility, then round-robins ONE item per video per
    round until ``n`` items are picked (or the pool is exhausted). No RNG
    involved — same input always yields the same sample, and a small ``n``
    naturally spreads across as many distinct videos as possible instead of
    draining one video first.
    """
    if n <= 0 or not items:
        return []
    by_video: Dict[str, List[Any]] = {}
    for it in items:
        vid = _get(it, video_key)
        by_video.setdefault(vid, []).append(it)
    for vid in by_video:
        by_video[vid].sort(key=lambda it: (_get(it, video_key), _get(it, id_key)))
    video_ids = sorted(by_video)
    queues = [list(by_video[v]) for v in video_ids]

    out: List[Any] = []
    i = 0
    while len(out) < n and any(queues):
        q = queues[i % len(queues)]
        if q:
            out.append(q.pop(0))
        i += 1
    return out[:n]


# Backward/forward-compatible alias — events use the exact same round-robin
# scheme, just with a different id column (event_id).
def sample_events(items: Sequence[Any], n: int) -> List[Any]:
    return sample_segments(items, n, id_key="event_id", video_key="video_id")


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------
def build_caption_judge_payload(
    caption: Any,
    prompts_dir: Optional[Path] = None,
) -> str:
    """Fill the caption-quality judge prompt for one (clip, caption) pair."""
    template = load_prompt_template(
        prompts_dir or _PROMPTS_DIR, CAPTION_JUDGE_PROMPT_FILE
    )
    prompt = template
    prompt = prompt.replace("{video_id}", str(_get(caption, "video_id")))
    prompt = prompt.replace("{segment_id}", str(_get(caption, "segment_id")))
    prompt = prompt.replace("{time_range}", str(_get(caption, "time_range", [])))
    prompt = prompt.replace(
        "{visual_description}", str(_get(caption, "visual_description", ""))
    )
    prompt = prompt.replace(
        "{audio_description}", str(_get(caption, "audio_description", ""))
    )
    return prompt


def build_event_judge_payload(
    event: Any,
    prompts_dir: Optional[Path] = None,
) -> str:
    """Fill the emotion-event judge prompt for one (clip, event) pair."""
    template = load_prompt_template(
        prompts_dir or _PROMPTS_DIR, EVENT_JUDGE_PROMPT_FILE
    )
    prompt = template
    prompt = prompt.replace("{video_id}", str(_get(event, "video_id")))
    prompt = prompt.replace("{event_id}", str(_get(event, "event_id")))
    prompt = prompt.replace("{time_range}", str(_get(event, "time_range", [])))
    prompt = prompt.replace(
        "{emotion_label}", str(_get(event, "emotion_label", ""))
    )
    prompt = prompt.replace(
        "{event_description}", str(_get(event, "event_description", ""))
    )
    prompt = prompt.replace(
        "{visual_evidence}", str(_get(event, "visual_evidence", []))
    )
    prompt = prompt.replace(
        "{audio_evidence}", str(_get(event, "audio_evidence", []))
    )
    return prompt


# ---------------------------------------------------------------------------
# Verdict parsing — tolerant of malformed judge JSON; never raises. A missing
# or out-of-range dimension yields score=None (excluded from aggregation) and
# sets a non-empty "parse_error" on the verdict so the CLI can log + continue.
# ---------------------------------------------------------------------------
def _extract_score_reason(entry: Any, lo: int, hi: int) -> tuple:
    if not isinstance(entry, dict):
        return None, ""
    reason = entry.get("reason", "")
    if not isinstance(reason, str):
        reason = str(reason)
    raw_score = entry.get("score")
    try:
        score = int(raw_score)
    except (TypeError, ValueError):
        return None, reason
    if not (lo <= score <= hi):
        return None, reason
    return score, reason


def parse_caption_verdict(raw: Any) -> Dict[str, Any]:
    """Parse a raw caption-judge response into ``{dim: {score, reason}, ...}``.

    Falls back to ``score=None`` per-dimension (never crashes) and sets
    ``parse_error`` to the first problem found, so a malformed/partial response
    degrades gracefully instead of aborting the whole judging run.
    """
    error: Optional[str] = None
    if not isinstance(raw, dict):
        error = "response is not a JSON object"
        raw = {}
    out: Dict[str, Any] = {}
    for dim in ("faithfulness", "hallucination", "coverage", "fluency"):
        score, reason = _extract_score_reason(raw.get(dim), 1, 5)
        if score is None:
            error = error or f"missing/invalid dimension: {dim}"
        out[dim] = {"score": score, "reason": reason}
    score, reason = _extract_score_reason(raw.get("emotion_leakage_ok"), 0, 1)
    if score is None:
        error = error or "missing/invalid dimension: emotion_leakage_ok"
    out["emotion_leakage_ok"] = {"score": score, "reason": reason}
    out["parse_error"] = error
    return out


def parse_event_verdict(raw: Any, assigned_label: str) -> Dict[str, Any]:
    """Parse a raw emotion-event judge response.

    ``label_agreement`` is NOT read from the model's own boolean — the model
    only reports its independently-picked ``predicted_label``; agreement
    (0/1) is computed HERE by comparing it (case/whitespace-insensitive) to
    ``assigned_label``, so the decision is derived in code rather than
    trusted from the model, consistent with ``verification.py``.
    """
    error: Optional[str] = None
    if not isinstance(raw, dict):
        error = "response is not a JSON object"
        raw = {}
    out: Dict[str, Any] = {}
    for dim in ("cue_sufficiency", "cue_grounded"):
        score, reason = _extract_score_reason(raw.get(dim), 1, 5)
        if score is None:
            error = error or f"missing/invalid dimension: {dim}"
        out[dim] = {"score": score, "reason": reason}

    la = raw.get("label_agreement")
    reason = ""
    predicted_label = None
    if isinstance(la, dict):
        reason = la.get("reason", "")
        if not isinstance(reason, str):
            reason = str(reason)
        pl = la.get("predicted_label")
        if isinstance(pl, str) and pl.strip():
            predicted_label = pl.strip()
    if predicted_label is None:
        error = error or "missing/invalid dimension: label_agreement"
        score = None
    else:
        score = 1 if predicted_label.lower() == (assigned_label or "").strip().lower() else 0
    out["label_agreement"] = {
        "score": score, "reason": reason, "predicted_label": predicted_label,
    }
    out["parse_error"] = error
    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def aggregate(verdicts: Sequence[Dict[str, Any]], dimensions: Sequence[str]) -> Dict[str, Any]:
    """Per-dimension mean + n (over non-None scores only) across ``verdicts``.

    ``n_items`` is the total number of verdicts fed in; ``n_errors`` counts
    verdicts that carried a non-empty ``parse_error`` (partial or total parse
    failure) so a run's reliability is visible alongside the scores.
    """
    per_dim: Dict[str, Any] = {}
    for dim in dimensions:
        scores = [
            v[dim]["score"] for v in verdicts
            if isinstance(v.get(dim), dict) and v[dim].get("score") is not None
        ]
        per_dim[dim] = {
            "mean": (sum(scores) / len(scores)) if scores else None,
            "n": len(scores),
        }
    n_errors = sum(1 for v in verdicts if v.get("parse_error"))
    return {"n_items": len(verdicts), "n_errors": n_errors, "dimensions": per_dim}


def combined_caption_score(agg: Dict[str, Any]) -> Optional[float]:
    """Weighted mean of the 1-5 caption dimensions (see ``CAPTION_COMBINED_WEIGHTS``).

    ``emotion_leakage_ok`` is intentionally excluded (0/1 scale, reported
    separately). Returns ``None`` if no weighted dimension has any score.
    """
    dims = agg.get("dimensions", {})
    total = 0.0
    total_w = 0.0
    for dim, w in CAPTION_COMBINED_WEIGHTS.items():
        mean = dims.get(dim, {}).get("mean")
        if mean is None:
            continue
        total += w * mean
        total_w += w
    return (total / total_w) if total_w > 0 else None
