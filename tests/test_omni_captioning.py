"""Local, model-free tests for the Qwen3-Omni captioning backend.

These never import vllm/torch/transformers/qwen_omni_utils and never load the
Qwen3-Omni model. They cover: prompt construction (one segment), robust JSON
extraction, required-field validation, the cache/resume decision, atomic write,
the OmniCaption -> EmotionCaption adapter, and that the heavy deps are not
imported at module import time.

Run:  python -m pytest tests/test_omni_captioning.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from emotion_query_pipeline import omni_captioning as oc
from emotion_query_pipeline.models import EMOTION_LABEL_VALUES, OmniCaption, Segment


def _segment(seg_id="s022", start=105.0, end=110.0, clip="clip.mp4") -> Segment:
    return Segment(
        segment_id=seg_id, index=22, start_time=start, end_time=end, clip_path=clip
    )


def _good_caption_dict(seg_id="s022", time_range=(105.0, 110.0)) -> dict:
    return {
        "segment_id": seg_id,
        "time_range": list(time_range),
        "visual_objective": {
            "people": [
                {
                    "person": "a woman in a white shirt",
                    "visibility": "clearly visible",
                    "position": "foreground",
                    "action": "talking to a man",
                }
            ],
            "scene": {"location": "indoor room", "setting": "conversation scene"},
            "objects": [],
            "interactions": [],
            "key_actions": ["leaning forward"],
            "visibility_notes": "",
        },
        "visual_expression": [
            {
                "person": "the woman in a white shirt",
                "facial_cues": ["eyes widened", "mouth open"],
                "body_cues": [],
                "gaze": "looking toward the man",
            }
        ],
        "audio_description": "The woman speaks with an excited voice.",
        "emotion_description": "The woman appears surprised, with widened eyes.",
        "confidence": "high",
        "evidence_strength": "clear",
    }


# ---------------------------------------------------------------------------
# No heavy deps at import time
# ---------------------------------------------------------------------------
def test_no_heavy_imports_at_module_load():
    for mod in ("vllm", "torch", "transformers", "qwen_omni_utils"):
        assert mod not in sys.modules, f"{mod} must not be imported at module load"


def test_constructing_captioner_does_not_load_model():
    cap = oc.Qwen3OmniCaptioner()  # cheap: stores config only
    assert cap._llm is None and cap._processor is None
    assert cap.use_audio_in_video is True
    assert cap.sampling_params == {
        "temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_tokens": 2048,
    }
    for mod in ("vllm", "torch", "transformers", "qwen_omni_utils"):
        assert mod not in sys.modules


# ---------------------------------------------------------------------------
# Prompt construction — exactly one segment
# ---------------------------------------------------------------------------
def test_prompt_contains_single_segment_metadata():
    prompt = oc.build_omni_caption_prompt(_segment())
    assert "s022" in prompt
    assert "105.00" in prompt and "110.00" in prompt
    # A different segment's id must not leak in.
    assert "s023" not in prompt


# ---------------------------------------------------------------------------
# Robust JSON extraction
# ---------------------------------------------------------------------------
def test_extract_plain_json():
    data = oc.extract_caption_json(json.dumps(_good_caption_dict()))
    assert data["segment_id"] == "s022"


def test_extract_json_with_markdown_fence_and_prose():
    raw = (
        "Sure, here is the caption:\n```json\n"
        + json.dumps(_good_caption_dict())
        + "\n```\nHope that helps!"
    )
    data = oc.extract_caption_json(raw)
    assert data["confidence"] == "high"


def test_extract_json_with_trailing_second_object():
    raw = json.dumps(_good_caption_dict()) + "\n{ \"junk\": 1 }"
    data = oc.extract_caption_json(raw)
    assert data["segment_id"] == "s022"


def test_extract_json_failure_raises():
    with pytest.raises(oc.CaptionParseError) as ei:
        oc.extract_caption_json("no json here at all")
    assert ei.value.reason == "json_parse_error"


# ---------------------------------------------------------------------------
# Required-field validation
# ---------------------------------------------------------------------------
def test_missing_required_fields_detected():
    d = _good_caption_dict()
    del d["emotion_description"]
    assert "emotion_description" in oc.missing_required_fields(d)


def test_bad_time_range_flagged():
    d = _good_caption_dict()
    d["time_range"] = [105.0]  # wrong length
    assert "time_range" in oc.missing_required_fields(d)


def test_complete_caption_has_no_missing_fields():
    assert oc.missing_required_fields(_good_caption_dict()) == []


# ---------------------------------------------------------------------------
# parse_caption overrides metadata from the trusted segment
# ---------------------------------------------------------------------------
def test_parse_caption_overrides_metadata():
    d = _good_caption_dict(seg_id="WRONG", time_range=(0.0, 1.0))
    cap = oc.parse_caption(json.dumps(d), _segment(), "vid01")
    assert cap.segment_id == "s022"  # forced from segment, not model echo
    assert cap.time_range == [105.0, 110.0]
    assert cap.video_id == "vid01"


def test_parse_caption_missing_fields_raises_with_raw():
    d = _good_caption_dict()
    del d["audio_description"]
    with pytest.raises(oc.CaptionParseError) as ei:
        oc.parse_caption(json.dumps(d), _segment(), "vid01")
    assert ei.value.reason == "missing_required_fields"
    assert ei.value.raw_text  # raw kept for debugging


# ---------------------------------------------------------------------------
# Adapter: OmniCaption -> EmotionCaption
# ---------------------------------------------------------------------------
def test_adapter_maps_fields_and_emotion_label():
    cap = OmniCaption.model_validate(_good_caption_dict())
    ec = oc.omni_to_emotion_caption(cap, "vid01")
    assert ec.segment_ids == ["s022"]
    assert ec.caption_id == "vid01_s022"
    assert "woman in a white shirt" in ec.person
    assert ec.sound == "The woman speaks with an excited voice."
    assert ec.emotion == "surprised"  # scanned from emotion_description
    assert "eyes widened" in ec.observable_evidence
    assert ec.confidence == "high" and ec.evidence_strength == "clear"


def test_adapter_unknown_emotion_defaults_neutral():
    d = _good_caption_dict()
    d["emotion_description"] = "It is unclear what they feel."
    ec = oc.omni_to_emotion_caption(OmniCaption.model_validate(d), "vid01")
    assert ec.emotion == "neutral"
    assert ec.emotion not in EMOTION_LABEL_VALUES  # so the filter drops it


# ---------------------------------------------------------------------------
# Atomic write + cache read
# ---------------------------------------------------------------------------
def test_atomic_write_and_read_back(tmp_path):
    path = oc.caption_cache_path(tmp_path, "vid01", "s022")
    oc.atomic_write_json(path, _good_caption_dict())
    assert path.exists()
    assert not path.with_suffix(".json.tmp").exists()  # tmp cleaned up
    cap, reason = oc.read_valid_cache(path)
    assert reason is None and cap.segment_id == "s022"


def test_read_valid_cache_missing(tmp_path):
    cap, reason = oc.read_valid_cache(tmp_path / "nope.json")
    assert cap is None and reason == "not_found"


def test_read_valid_cache_corrupt_json(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{ not json", encoding="utf-8")
    cap, reason = oc.read_valid_cache(p)
    assert cap is None and reason == "json_parse_error"


def test_read_valid_cache_missing_fields(tmp_path):
    d = _good_caption_dict()
    del d["visual_objective"]
    p = tmp_path / "incomplete.json"
    p.write_text(json.dumps(d), encoding="utf-8")
    cap, reason = oc.read_valid_cache(p)
    assert cap is None and reason == "missing_required_fields"


# ---------------------------------------------------------------------------
# Resume / cache behaviour with a fake captioner (no model)
# ---------------------------------------------------------------------------
class FakeCaptioner:
    """Returns a canned JSON string and counts how many times it was called."""

    def __init__(self, payload: dict):
        self.payload = payload
        self.calls = 0

    def caption(self, prompt_text: str, clip_path: str) -> str:
        self.calls += 1
        return "```json\n" + json.dumps(self.payload) + "\n```"


def test_generate_then_resume_skips_model(tmp_path):
    cache, raw = tmp_path / "captions", tmp_path / "raw"
    fake = FakeCaptioner(_good_caption_dict())
    seg = _segment()

    # First pass: model is called, cache written.
    out = oc.caption_video_omni("vid01", [seg], fake, cache, raw)
    assert fake.calls == 1 and len(out) == 1
    assert oc.caption_cache_path(cache, "vid01", "s022").exists()

    # Second pass (resume): cache hit, model NOT called again.
    out2 = oc.caption_video_omni("vid01", [seg], fake, cache, raw)
    assert fake.calls == 1  # unchanged
    assert out2[0].segment_id == "s022"


def test_overwrite_forces_regeneration(tmp_path):
    cache, raw = tmp_path / "captions", tmp_path / "raw"
    fake = FakeCaptioner(_good_caption_dict())
    seg = _segment()
    oc.caption_video_omni("vid01", [seg], fake, cache, raw)
    assert fake.calls == 1
    oc.caption_video_omni("vid01", [seg], fake, cache, raw, overwrite=True)
    assert fake.calls == 2  # regenerated despite valid cache


def test_invalid_cache_triggers_regeneration(tmp_path):
    cache, raw = tmp_path / "captions", tmp_path / "raw"
    # Pre-seed a corrupt cache file.
    path = oc.caption_cache_path(cache, "vid01", "s022")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ corrupt", encoding="utf-8")
    fake = FakeCaptioner(_good_caption_dict())
    oc.caption_video_omni("vid01", [_segment()], fake, cache, raw)
    assert fake.calls == 1  # regenerated because cache was invalid
    cap, reason = oc.read_valid_cache(path)
    assert reason is None  # now valid


def test_parse_failure_saves_raw_and_skips(tmp_path):
    cache, raw = tmp_path / "captions", tmp_path / "raw"

    class BadCaptioner:
        def caption(self, prompt_text, clip_path):
            return "the model rambled and produced no json"

    out = oc.caption_video_omni("vid01", [_segment()], BadCaptioner(), cache, raw)
    assert out == []  # segment skipped, video not aborted
    assert oc.raw_output_path(raw, "vid01", "s022").exists()  # raw saved for debug
