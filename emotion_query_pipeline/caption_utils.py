"""Shared, model-agnostic caption helpers: JSON extraction, validation, salvage,
and the per-segment cache / resume path.

Used by the observation caption backend(s) and by the Qwen3-Omni verify/rewrite
client. Pure Python — no heavy deps, no GPU — so it imports and tests locally.
Captions are OBSERVATION-ONLY (no emotion); emotion is judged later in the Gemini
emotion-event stage.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import OMNI_REQUIRED_FIELDS, OmniCaption, Segment


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class CaptionParseError(Exception):
    """Raised when a raw model output can't be turned into a valid OmniCaption.

    ``reason`` is a short machine code used in logs / cache-invalidation messages:
    ``json_parse_error``, ``missing_required_fields`` or ``schema_validation_error``.
    ``raw_text`` is the offending model output (for debugging / raw dump).
    """

    def __init__(self, reason: str, message: str, raw_text: str = "") -> None:
        super().__init__(f"{reason}: {message}")
        self.reason = reason
        self.raw_text = raw_text


# ---------------------------------------------------------------------------
# Robust JSON extraction + validation
# ---------------------------------------------------------------------------
def extract_caption_json(raw_text: str) -> dict:
    """Pull the first JSON object out of a raw model response.

    Tolerates markdown fences (```json ... ```), leading/trailing prose, and a
    second trailing JSON value. Raises ``CaptionParseError(json_parse_error)``
    if no JSON object can be decoded.
    """
    if raw_text is None:
        raise CaptionParseError("json_parse_error", "empty (None) response", "")
    text = raw_text.strip()
    if not text:
        raise CaptionParseError("json_parse_error", "empty response", raw_text)

    # Strip a fenced block if present (```json ... ``` or ``` ... ```).
    if "```" in text:
        fence = text.find("```")
        rest = text[fence + 3 :]
        if "\n" in rest:
            rest = rest.split("\n", 1)[1]  # drop the ```json language tag line
        end = rest.rfind("```")
        if end != -1:
            rest = rest[:end]
        text = rest.strip()

    # Locate the first '{' and decode one JSON value, ignoring trailing junk.
    start = text.find("{")
    if start == -1:
        raise CaptionParseError(
            "json_parse_error", "no JSON object found in output", raw_text
        )
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError as e:
        raise CaptionParseError("json_parse_error", str(e), raw_text) from e
    if not isinstance(obj, dict):
        raise CaptionParseError(
            "json_parse_error", "top-level JSON is not an object", raw_text
        )
    return obj


def missing_required_fields(data: dict) -> List[str]:
    """Required top-level observation fields that are absent or empty.

    ``time_range`` must be a 2-element list; the other required fields just have to
    be present and non-None (empty strings/lists are allowed so a genuinely
    sparse-but-complete caption still validates). ``temporal_description`` is
    optional and not checked here.
    """
    missing: List[str] = []
    for fld in OMNI_REQUIRED_FIELDS:
        if fld not in data or data[fld] is None:
            missing.append(fld)
            continue
        if fld == "time_range":
            tr = data[fld]
            if not isinstance(tr, (list, tuple)) or len(tr) != 2:
                missing.append(fld)
    return missing


def salvage_caption(
    item: Optional[dict], raw_text: str, segment: Segment, video_id: str
) -> OmniCaption:
    """Best-effort OmniCaption for a segment whose output failed strict parsing.

    Keeps whatever decoded (``OmniCaption`` allows extra keys and defaults the
    rest), forces the trusted metadata, and pins ``confidence=low`` /
    ``evidence_strength=weak`` so downstream treats it as soft evidence. When
    nothing decoded, the raw model text is stuffed into ``temporal_description``
    so the segment is at least described. Marked ``caption_status="salvaged"``.
    """
    tr = [round(segment.start_time, 2), round(segment.end_time, 2)]
    data: Dict[str, Any] = dict(item) if isinstance(item, dict) else {}
    data["segment_id"] = segment.segment_id
    data["time_range"] = tr
    data["video_id"] = video_id
    data["caption_status"] = "salvaged"
    if data.get("confidence") not in ("high", "medium", "low"):
        data["confidence"] = "low"
    data["evidence_strength"] = "weak"
    fallback = (raw_text or "").strip()[:500] or "(unparseable)"
    if not (str(data.get("audio_description") or "")).strip() and not (
        str(data.get("temporal_description") or "")
    ).strip():
        data["temporal_description"] = fallback
    try:
        return OmniCaption.model_validate(data)
    except Exception:
        # Last resort: a minimal valid caption keeping only trusted text so the
        # segment is NEVER dropped, whatever odd shape the nested fields had.
        audio = data.get("audio_description")
        return OmniCaption(
            segment_id=segment.segment_id,
            video_id=video_id,
            time_range=tr,
            audio_description=audio if isinstance(audio, str) else "",
            temporal_description=fallback,
            confidence="low",
            evidence_strength="weak",
            caption_status="salvaged",
        )


# ---------------------------------------------------------------------------
# Cache / resume helpers (atomic write, raw dump)
# ---------------------------------------------------------------------------
def caption_cache_path(cache_dir: Path, video_id: str, segment_id: str) -> Path:
    return Path(cache_dir) / video_id / f"{segment_id}.json"


def raw_output_path(raw_dir: Path, video_id: str, segment_id: str) -> Path:
    return Path(raw_dir) / video_id / f"{segment_id}.txt"


def atomic_write_json(path: Path, data: dict) -> None:
    """Write ``data`` as JSON to ``path`` atomically (tmp file + flush + rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)  # atomic on POSIX/Windows for same-filesystem rename


def save_raw_output(
    raw_dir: Path, video_id: str, segment_id: str, raw_text: str, reason: str
) -> Path:
    """Persist a failed model output for debugging. Returns the file path."""
    path = raw_output_path(raw_dir, video_id, segment_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = f"# parse failure: reason={reason}\n# segment_id={segment_id}\n\n"
    path.write_text(header + (raw_text or ""), encoding="utf-8")
    return path


def read_valid_cache(path: Path):
    """Load a cached caption if it is present, parseable and complete.

    Returns ``(OmniCaption, None)`` on a cache hit, or ``(None, reason)`` where
    ``reason`` is ``"not_found"``, ``"json_parse_error"``,
    ``"missing_required_fields"`` or ``"schema_validation_error"``.
    """
    path = Path(path)
    if not path.exists():
        return None, "not_found"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None, "json_parse_error"
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None, "json_parse_error"
    if not isinstance(data, dict):
        return None, "json_parse_error"
    missing = missing_required_fields(data)
    if missing:
        return None, "missing_required_fields"
    try:
        return OmniCaption.model_validate(data), None
    except Exception:
        return None, "schema_validation_error"


def _chunks(items: list, size: int) -> List[list]:
    return [items[i : i + size] for i in range(0, len(items), max(1, size))]


def _resolve_cache(
    video_id: str,
    segments: List[Segment],
    cache_dir: Path,
    resume: bool,
    overwrite: bool,
):
    """Split segments into (cached results, segments needing generation).

    Logs skip / regenerate decisions. Segments without a clip are dropped.
    """
    cached: Dict[str, OmniCaption] = {}
    to_generate: List[Segment] = []
    for segment in segments:
        if not segment.clip_path:
            continue
        if resume and not overwrite:
            cap, reason = read_valid_cache(
                caption_cache_path(cache_dir, video_id, segment.segment_id)
            )
            if cap is not None:
                print(f"[caption] skip existing: video_id={video_id} "
                      f"segment_id={segment.segment_id}")
                cached[segment.segment_id] = cap
                continue
            if reason != "not_found":
                print(f"[caption] regenerate invalid cache: video_id={video_id} "
                      f"segment_id={segment.segment_id} reason={reason}")
        to_generate.append(segment)
    return cached, to_generate
