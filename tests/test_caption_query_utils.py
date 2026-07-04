"""Offline unit tests for the caption→query integration utilities.

These NEVER load a caption model and NEVER call Gemini: they exercise the pure
normalization / boundary / downstream-plumbing helpers with fake data, and assert
that importing the new module/script pulls in none of the heavy deps.

Run:  python -m pytest tests/test_caption_query_utils.py -q
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from emotion_query_pipeline import caption_query_test as cqt
from emotion_query_pipeline.models import (
    EmotionEventOutput,
    OmniCaption,
    Segment,
)

_HEAVY = ("torch", "transformers", "qwen_omni_utils", "qwen_vl_utils", "decord", "soundfile")


def _seg(seg_id="s001", start=0.0, end=5.0, clip="clip.mp4") -> Segment:
    return cqt.make_segment(segment_id=seg_id, start=start, end=end, clip_path=clip)


# ---------------------------------------------------------------------------
# No heavy imports when importing the module or the CLI script
# ---------------------------------------------------------------------------
def test_importing_module_does_not_import_heavy_deps():
    for m in _HEAVY:
        sys.modules.pop(m, None)
    importlib.reload(cqt)
    for m in _HEAVY:
        assert m not in sys.modules, f"{m} must not be imported by caption_query_test"


def test_importing_cli_script_does_not_import_heavy_deps():
    for m in _HEAVY:
        sys.modules.pop(m, None)
    importlib.import_module("run_caption_query_test")
    for m in _HEAVY:
        assert m not in sys.modules, f"{m} must not be imported by the CLI script"


# ---------------------------------------------------------------------------
# normalize_to_omni_caption
# ---------------------------------------------------------------------------
def test_normalize_plain_text_av():
    seg = _seg()
    cap = cqt.normalize_to_omni_caption(
        "A woman turns quickly and her eyes widen.", seg, "v",
        source_caption_model="qwen3_omni", modality="av",
    )
    assert isinstance(cap, OmniCaption)
    assert cap.segment_id == "s001" and cap.video_id == "v"
    assert cap.time_range == [0.0, 5.0]
    # AV plain text lands in temporal_description; visuals not fabricated.
    assert "eyes widen" in cap.temporal_description
    assert cap.source_caption_model == "qwen3_omni"


def test_normalize_json_maps_fields():
    seg = _seg()
    raw = {
        "segment_id": "WRONG", "time_range": [99, 100],  # must be ignored
        "visual_objective": {"people": [{"person": "a man", "action": "standing"}]},
        "visual_expression": [{"person": "a man", "facial_cues": ["frown"]}],
        "audio_description": "a raised voice",
        "temporal_description": "voice rises",
        "confidence": "high", "evidence_strength": "clear",
    }
    cap = cqt.normalize_to_omni_caption(
        raw, seg, "v", source_caption_model="avocado", modality="av",
    )
    # Trusted metadata forced from the segment, model echo discarded.
    assert cap.segment_id == "s001" and cap.time_range == [0.0, 5.0]
    assert cap.visual_objective.people[0].person == "a man"
    assert cap.visual_expression[0].facial_cues == ["frown"]
    assert cap.audio_description == "a raised voice"
    assert cap.confidence == "high" and cap.evidence_strength == "clear"
    assert cap.caption_status == "normalized"


def test_normalize_json_string_with_fence():
    seg = _seg()
    raw = '```json\n{"audio_description": "laughing", "confidence": "medium"}\n```'
    cap = cqt.normalize_to_omni_caption(
        raw, seg, "v", source_caption_model="x", modality="av",
    )
    assert cap.audio_description == "laughing" and cap.confidence == "medium"


@pytest.mark.parametrize("raw", ["", None, "   ", "%%%not json%%%{oops"])
def test_normalize_malformed_or_empty_salvages(raw):
    seg = _seg()
    cap = cqt.normalize_to_omni_caption(
        raw, seg, "v", source_caption_model="x", modality="av",
    )
    assert isinstance(cap, OmniCaption)
    assert cap.confidence == "low" and cap.evidence_strength == "weak"
    assert cap.caption_status == "salvaged"
    assert cap.segment_id == "s001"


def test_normalize_audio_only_never_fabricates_visual():
    seg = _seg()
    cap = cqt.normalize_to_omni_caption(
        "a trembling, shaky voice", seg, "v",
        source_caption_model="secap", modality="audio",
    )
    assert cap.audio_description == "a trembling, shaky voice"
    assert not cap.temporal_description
    assert cap.visual_objective.people == [] and cap.visual_expression == []


def test_normalize_video_only_never_fabricates_audio():
    seg = _seg()
    cap = cqt.normalize_to_omni_caption(
        "a man paces the room", seg, "v",
        source_caption_model="qwen_vl", modality="video",
    )
    assert cap.audio_description == ""  # video-only must not invent audio
    assert "paces" in cap.temporal_description


# ---------------------------------------------------------------------------
# merge_audio_video_caption
# ---------------------------------------------------------------------------
def test_merge_audio_video_caption():
    seg = _seg()
    cap = cqt.merge_audio_video_caption(
        audio_text="a shaky, tearful voice",
        video_text="a woman covers her face with both hands",
        segment=seg, video_id="v",
        audio_source_model="yaoxunxu/SECaps",
        video_source_model="Qwen/Qwen3-VL-8B-Instruct",
        source_caption_model="secap_qwen",
    )
    assert isinstance(cap, OmniCaption)
    # Audio evidence from the audio model; visual/temporal from the video model.
    assert cap.audio_description == "a shaky, tearful voice"
    assert "covers her face" in cap.temporal_description
    assert cap.audio_source_model == "yaoxunxu/SECaps"
    assert cap.video_source_model == "Qwen/Qwen3-VL-8B-Instruct"


# ---------------------------------------------------------------------------
# Input boundary validation
# ---------------------------------------------------------------------------
def test_qwen3_omni_missing_video_errors():
    with pytest.raises(ValueError, match="requires --video"):
        cqt.validate_inputs("qwen3_omni", video=None, audio=None)


@pytest.mark.parametrize("model", ["qwen_audio_vl", "af3_vl", "secap_qwen"])
def test_audio_video_models_require_both(model):
    with pytest.raises(ValueError, match="--video"):
        cqt.validate_inputs(model, video=None, audio="a.wav")
    with pytest.raises(ValueError, match="--audio"):
        cqt.validate_inputs(model, video="v.mp4", audio=None)


def test_validate_inputs_ok_returns_spec():
    spec = cqt.validate_inputs("qwen_audio_vl", video="v.mp4", audio="a.wav")
    assert spec.name == "qwen_audio_vl" and spec.requires_audio


def test_unknown_model_errors():
    with pytest.raises(ValueError, match="unknown"):
        cqt.validate_inputs("not_a_model", video="v.mp4", audio=None)


# ---------------------------------------------------------------------------
# secap_qwen must use SECap for audio and must NOT reference Qwen3-Omni-Captioner
# ---------------------------------------------------------------------------
def test_secap_qwen_uses_secap_not_omni_captioner():
    spec = cqt.get_model_spec("secap_qwen")
    assert "SECaps" in spec.default_audio_model_path
    assert "Qwen3-Omni" not in spec.default_audio_model_path
    assert "Captioner" not in spec.default_audio_model_path
    assert spec.default_video_model_path == "Qwen/Qwen3-VL-8B-Instruct"


def test_af3_marked_non_commercial():
    assert cqt.get_model_spec("af3_vl").non_commercial is True


# ---------------------------------------------------------------------------
# build_downstream_inputs + run_downstream_gemini with a FAKE client
# ---------------------------------------------------------------------------
def _omni(seg_id, start, end):
    return OmniCaption.model_validate({
        "segment_id": seg_id, "time_range": [start, end],
        "visual_objective": {"people": [{"person": "a woman", "action": "talking"}]},
        "visual_expression": [{"person": "a woman", "facial_cues": ["eyes widened"]}],
        "audio_description": "a raised voice", "temporal_description": "voice rises",
        "confidence": "high", "evidence_strength": "clear",
    })


class FakeDownstreamClient:
    """BaseLLMClient-compatible fake: returns canned events then queries."""

    def __init__(self, events, queries):
        self._events, self._queries = events, queries

    def generate_json(self, prompt, schema_name, video_uri=None):
        if schema_name == "EmotionEventOutput":
            return {"events": list(self._events)}
        if schema_name == "GenerationOutput":
            return {"queries": list(self._queries)}
        raise AssertionError(f"unexpected schema {schema_name}")


def test_build_downstream_inputs_type_checks():
    seg = _seg()
    inp = cqt.build_downstream_inputs("v", [_omni("s001", 0.0, 5.0)], [seg])
    assert inp.video_id == "v"
    assert isinstance(inp.captions[0], OmniCaption)
    assert isinstance(inp.segments[0], Segment)
    with pytest.raises(TypeError):
        cqt.build_downstream_inputs("v", [{"not": "a caption"}], [seg])


def test_run_downstream_gemini_with_fake_client_produces_queries():
    seg = _seg()
    caps = [_omni("s001", 0.0, 5.0)]
    client = FakeDownstreamClient(
        events=[{
            "event_id": "v_e01", "emotion_label": "surprised",
            "event_description": "the woman widens her eyes", "time_range": [0.0, 5.0],
            "visual_evidence": ["eyes widened"], "confidence": "high",
            "evidence_strength": "clear",
        }],
        queries=[{
            "query_type": "emotion_state",
            "query_text": "When does the woman appear surprised?",
            "time_range": [1.0, 4.0],
        }],
    )
    inp = cqt.build_downstream_inputs("v", caps, [seg])
    out = cqt.run_downstream_gemini(inp, client)
    assert isinstance(out["events"], EmotionEventOutput)
    assert len(out["events"].events) == 1
    assert len(out["generation"].queries) == 1
    assert out["generation"].queries[0].segment_ids == ["s001"]
    assert out["warnings"] == []  # events + queries both present


def test_run_downstream_gemini_zero_queries_warns():
    seg = _seg()
    caps = [_omni("s001", 0.0, 5.0)]
    client = FakeDownstreamClient(events=[], queries=[])
    inp = cqt.build_downstream_inputs("v", caps, [seg])
    out = cqt.run_downstream_gemini(inp, client)
    assert out["generation"].queries == []
    assert any("0 events" in w for w in out["warnings"])
    assert any("0 queries" in w for w in out["warnings"])


# ---------------------------------------------------------------------------
# save_outputs writes every artefact
# ---------------------------------------------------------------------------
def test_save_outputs_writes_all_files(tmp_path):
    from emotion_query_pipeline.models import EmotionEventOutput, GenerationOutput
    seg = _seg()
    caps = [_omni("s001", 0.0, 5.0)]
    written = cqt.save_outputs(
        tmp_path,
        raw_records=[{"segment_id": "s001", "raw_output": "text"}],
        captions=caps,
        events=EmotionEventOutput(video_id="v", events=[]),
        generation=GenerationOutput(video_id="v", queries=[]),
        metadata={"caption_model": "qwen3_omni", "warnings": []},
    )
    for name in (
        "raw_caption_output.json", "normalized_captions.jsonl",
        "emotion_events.json", "generated_queries.json", "run_metadata.json",
    ):
        assert (tmp_path / name).is_file(), name
        assert name in written
    # final_queries.json only when provided
    assert not (tmp_path / "final_queries.json").exists()
