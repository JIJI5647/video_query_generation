"""Local, model-free tests for the shared caption helpers (caption_utils).

Cover robust JSON extraction, required-field validation, salvage, atomic write,
cache read, and the resume decision. Captions are observation-only (no emotion).

Run:  python -m pytest tests/test_caption_utils.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from emotion_query_pipeline import caption_utils as cu
from emotion_query_pipeline.models import OmniCaption, Segment


def _segment(seg_id="s022", index=22, start=105.0, end=110.0, clip="clip.mp4") -> Segment:
    return Segment(
        segment_id=seg_id, index=index, start_time=start, end_time=end, clip_path=clip
    )


def _segments(n):
    return [
        _segment(seg_id=f"s{i:03d}", index=i, start=i * 5.0, end=i * 5.0 + 5.0,
                 clip=f"c{i}.mp4")
        for i in range(1, n + 1)
    ]


def _good_caption_dict(seg_id="s022", time_range=(105.0, 110.0)) -> dict:
    return {
        "segment_id": seg_id,
        "time_range": list(time_range),
        "visual_objective": {
            "people": [{
                "person": "a woman in a white shirt",
                "visibility": "clearly visible",
                "position": "foreground",
                "action": "talking to a man",
            }],
            "scene": {"location": "indoor room", "setting": "conversation scene"},
            "objects": [], "interactions": [], "key_actions": ["leaning forward"],
            "visibility_notes": "",
        },
        "visual_expression": [{
            "person": "the woman in a white shirt",
            "facial_cues": ["eyes widened", "mouth open"],
            "body_cues": [], "gaze": "looking toward the man",
        }],
        "audio_description": "The woman speaks with an excited voice.",
        "temporal_description": "Her voice rises sharply midway through the clip.",
        "confidence": "high",
        "evidence_strength": "clear",
    }


# --- JSON extraction --------------------------------------------------------
def test_extract_object_plain_and_fenced():
    assert cu.extract_caption_json(json.dumps(_good_caption_dict()))["segment_id"] == "s022"
    fenced = "ok:\n```json\n" + json.dumps(_good_caption_dict()) + "\n```\n"
    assert cu.extract_caption_json(fenced)["confidence"] == "high"


def test_extract_failure_raises():
    with pytest.raises(cu.CaptionParseError):
        cu.extract_caption_json("no json")


# --- required-field validation ---------------------------------------------
def test_missing_required_fields():
    d = _good_caption_dict(); del d["audio_description"]
    assert "audio_description" in cu.missing_required_fields(d)
    d2 = _good_caption_dict(); d2["time_range"] = [1.0]
    assert "time_range" in cu.missing_required_fields(d2)
    # temporal_description is OPTIONAL — its absence must not flag missing.
    d3 = _good_caption_dict(); d3.pop("temporal_description", None)
    assert cu.missing_required_fields(d3) == []
    assert cu.missing_required_fields(_good_caption_dict()) == []


# --- observation-only: no emotion field ------------------------------------
def test_caption_has_no_emotion_field():
    cap = OmniCaption.model_validate(_good_caption_dict())
    assert not hasattr(cap, "emotion_description")
    assert cap.temporal_description.startswith("Her voice rises")


# --- salvage ----------------------------------------------------------------
def test_salvage_keeps_partial_and_forces_metadata():
    seg = _segment()
    cap = cu.salvage_caption({"audio_description": "shouting"}, "raw text", seg, "vid01")
    assert cap.segment_id == "s022" and cap.video_id == "vid01"
    assert cap.confidence == "low" and cap.evidence_strength == "weak"
    assert getattr(cap, "caption_status", None) == "salvaged"
    assert cap.audio_description == "shouting"


def test_salvage_unparseable_stuffs_raw_into_temporal():
    seg = _segment()
    cap = cu.salvage_caption(None, "totally broken output", seg, "vid01")
    assert "broken" in cap.temporal_description


# --- cache read / write -----------------------------------------------------
def test_atomic_write_and_read_back(tmp_path):
    path = cu.caption_cache_path(tmp_path, "vid01", "s022")
    cu.atomic_write_json(path, _good_caption_dict())
    assert path.exists() and not path.with_suffix(".json.tmp").exists()
    cap, reason = cu.read_valid_cache(path)
    assert reason is None and cap.segment_id == "s022"


def test_read_valid_cache_states(tmp_path):
    assert cu.read_valid_cache(tmp_path / "nope.json") == (None, "not_found")
    bad = tmp_path / "bad.json"; bad.write_text("{ not json", encoding="utf-8")
    assert cu.read_valid_cache(bad)[1] == "json_parse_error"
    inc = tmp_path / "inc.json"
    d = _good_caption_dict(); del d["visual_objective"]
    inc.write_text(json.dumps(d), encoding="utf-8")
    assert cu.read_valid_cache(inc)[1] == "missing_required_fields"


# --- resume decision --------------------------------------------------------
def test_resolve_cache_skips_valid_and_queues_missing(tmp_path):
    segs = _segments(2)
    # Pre-cache s001 only.
    cu.atomic_write_json(
        cu.caption_cache_path(tmp_path, "v", "s001"), _good_caption_dict("s001")
    )
    cached, to_generate = cu._resolve_cache("v", segs, tmp_path, True, False)
    assert set(cached) == {"s001"}
    assert [s.segment_id for s in to_generate] == ["s002"]


def test_resolve_cache_overwrite_regenerates_all(tmp_path):
    segs = _segments(2)
    cu.atomic_write_json(
        cu.caption_cache_path(tmp_path, "v", "s001"), _good_caption_dict("s001")
    )
    cached, to_generate = cu._resolve_cache("v", segs, tmp_path, True, overwrite=True)
    assert cached == {}
    assert [s.segment_id for s in to_generate] == ["s001", "s002"]
