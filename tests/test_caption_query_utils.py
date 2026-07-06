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


@pytest.mark.parametrize("script", [
    "run_caption_generation_test", "run_query_generation_test", "run_evaluation_test",
    "run_caption_generation",
])
def test_importing_cli_script_does_not_import_heavy_deps(script):
    for m in _HEAVY:
        sys.modules.pop(m, None)
    importlib.import_module(script)
    for m in _HEAVY:
        assert m not in sys.modules, f"{m} must not be imported by {script}"


def test_importing_batch_captioning_does_not_import_heavy_deps():
    for m in _HEAVY:
        sys.modules.pop(m, None)
    importlib.import_module("emotion_query_pipeline.batch_captioning")
    for m in _HEAVY:
        assert m not in sys.modules, f"{m} must not be imported by batch_captioning"


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


def test_normalize_output_picks_confidence_by_caption_model():
    # _normalize_output (the CLI script's dispatcher) maps caption_model ->
    # default_confidence: avocado/timechat (purpose-built cross-modal
    # captioners, raw text passed through as-is) get "medium"; anything else
    # keeps the generic plain-text fallback's "low".
    rcqt = importlib.import_module("run_caption_generation_test")
    seg = _seg()
    for model, expected in (("avocado", "medium"), ("timechat", "medium"),
                            ("qwen3_omni", "low"), ("unknown_model", "low")):
        out = cqt.CaptionModelOutput(
            modality="av", raw_output="a fused audiovisual narrative.",
            source_caption_model=model,
        )
        cap = rcqt._normalize_output(out, seg, "v", model)
        assert cap.confidence == expected, f"{model} -> {cap.confidence}"


def test_normalize_plain_text_default_confidence_override():
    # AVoCaDO's fused AV narrative is trusted at a caller-chosen default
    # ("medium") rather than the generic plain-text fallback's "low", since it's
    # a purpose-built cross-modal captioner rather than a naive text dump.
    seg = _seg()
    cap = cqt.normalize_to_omni_caption(
        "She screams as the camera whip-pans to the doorway.", seg, "v",
        source_caption_model="avocado", modality="av", default_confidence="medium",
    )
    assert cap.confidence == "medium"
    assert cap.evidence_strength == "ambiguous"  # still not fabricated as "clear"
    assert cap.caption_status == "normalized"


def test_normalize_malformed_ignores_default_confidence():
    # Salvaged (empty/malformed) output must stay "low"/"weak" regardless of
    # default_confidence — a genuine failure is never upgraded.
    seg = _seg()
    cap = cqt.normalize_to_omni_caption(
        "", seg, "v", source_caption_model="avocado", modality="av",
        default_confidence="medium",
    )
    assert cap.confidence == "low" and cap.evidence_strength == "weak"
    assert cap.caption_status == "salvaged"


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
# normalize_caption_output (shared modality dispatch + per-model confidence)
# ---------------------------------------------------------------------------
def test_normalize_caption_output_av_confidence_by_model():
    seg = _seg()
    for model, expected in (("avocado", "medium"), ("timechat", "medium"),
                            ("qwen3_omni", "low"), ("unknown_model", "low")):
        out = cqt.CaptionModelOutput(
            modality="av", raw_output="a fused audiovisual narrative.",
            source_caption_model=model,
        )
        cap = cqt.normalize_caption_output(out, seg, "v", model)
        assert cap.confidence == expected, f"{model} -> {cap.confidence}"


def test_normalize_caption_output_audio_video_merges():
    seg = _seg()
    out = cqt.CaptionModelOutput(
        modality="audio_video", audio_text="a trembling voice",
        video_text="a man backs away", source_caption_model="qwen_audio_vl",
        audio_source_model="A", video_source_model="B",
    )
    cap = cqt.normalize_caption_output(out, seg, "v", "qwen_audio_vl")
    assert cap.audio_description == "a trembling voice"
    assert "backs away" in cap.temporal_description
    assert cap.audio_source_model == "A" and cap.video_source_model == "B"


# ---------------------------------------------------------------------------
# batch_captioning: factory routing + a fake-session batch round trip
# ---------------------------------------------------------------------------
def test_batch_session_factory_unknown_model_errors():
    from emotion_query_pipeline import batch_captioning as bc
    with pytest.raises(ValueError, match="unknown"):
        bc.build_caption_session("not_a_model", cqt.RunnerConfig())


def test_batch_fake_session_round_trip_av():
    # A fake session that never loads a model — proves the normalize + record
    # path the batch script relies on works end-to-end without any heavy deps.
    from emotion_query_pipeline import batch_captioning as bc

    class FakeAVSession:
        def __init__(self):
            self.closed = False
            self.calls = 0

        def caption(self, segment, video_path, audio_path):
            self.calls += 1
            return cqt.CaptionModelOutput(
                modality="av",
                raw_output=f"caption for {segment.segment_id}",
                source_caption_model="fake",
            )

        def close(self):
            self.closed = True

    sess = FakeAVSession()
    segs = [_seg("s001", 0.0, 5.0), _seg("s002", 5.0, 10.0)]
    caps = []
    try:
        for s in segs:
            out = sess.caption(s, s.clip_path, None)
            caps.append(cqt.normalize_caption_output(out, s, "vid", "avocado"))
    finally:
        sess.close()
    assert sess.calls == 2 and sess.closed
    assert [c.segment_id for c in caps] == ["s001", "s002"]
    assert "s002" in caps[1].temporal_description
    assert caps[0].confidence == "medium"  # avocado default
    # Both are valid OmniCaptions that round-trip through write/read helpers.
    assert all(isinstance(c, OmniCaption) for c in caps)
    # AudioVideoSession composite exists and exposes the session interface.
    assert hasattr(bc.AudioVideoSession, "caption")
    assert hasattr(bc.AudioVideoSession, "close")


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
# Stage writers/loaders: save_caption_outputs / save_generation_outputs /
# save_evaluation_outputs, and their matching load_* round trips.
# ---------------------------------------------------------------------------
def test_save_caption_outputs_writes_all_files_and_round_trips(tmp_path):
    seg = _seg()
    caps = [_omni("s001", 0.0, 5.0)]
    written = cqt.save_caption_outputs(
        tmp_path,
        raw_records=[{"segment_id": "s001", "raw_output": "text"}],
        captions=caps, segments=[seg],
        metadata={"caption_model": "qwen3_omni"},
    )
    for name in (
        "raw_caption_output.json", "normalized_captions.jsonl",
        "segments.jsonl", "run_metadata.json",
    ):
        assert (tmp_path / name).is_file(), name
        assert name in written

    reloaded_segments, reloaded_captions = cqt.load_caption_outputs(tmp_path)
    assert len(reloaded_segments) == 1 and isinstance(reloaded_segments[0], Segment)
    assert reloaded_segments[0].segment_id == "s001"
    assert reloaded_segments[0].clip_path == "clip.mp4"
    assert len(reloaded_captions) == 1 and isinstance(reloaded_captions[0], OmniCaption)
    assert reloaded_captions[0].segment_id == "s001"


def test_load_caption_outputs_missing_dir_returns_empty(tmp_path):
    segments, captions = cqt.load_caption_outputs(tmp_path / "does_not_exist")
    assert segments == [] and captions == []


def test_save_generation_outputs_writes_all_files_and_round_trips(tmp_path):
    from emotion_query_pipeline.models import EmotionEventOutput, GenerationOutput
    seg = _seg()
    generation = GenerationOutput(video_id="v", queries=[])
    written = cqt.save_generation_outputs(
        tmp_path,
        events=EmotionEventOutput(video_id="v", events=[]),
        generation=generation, segments=[seg],
        metadata={"num_generated_queries": 0, "warnings": ["query-generation stage produced 0 queries for these captions."]},
    )
    for name in (
        "emotion_events.json", "generated_queries.json",
        "segments.jsonl", "generation_metadata.json",
    ):
        assert (tmp_path / name).is_file(), name
        assert name in written

    reloaded_segments, reloaded_generation = cqt.load_generation_outputs(tmp_path)
    assert len(reloaded_segments) == 1
    assert isinstance(reloaded_generation, GenerationOutput)
    assert reloaded_generation.video_id == "v"


def test_save_evaluation_outputs_writes_all_files(tmp_path):
    written = cqt.save_evaluation_outputs(
        tmp_path,
        final_queries=[{"query_id": "q1", "final_status": "accepted"}],
        summary={"total": 1, "final_status": {"accepted": 1}},
        metadata={"num_queries_in": 1, "num_queries_checked": 1},
    )
    for name in (
        "final_queries.json", "verification_summary.json", "evaluation_metadata.json",
    ):
        assert (tmp_path / name).is_file(), name
        assert name in written


# ---------------------------------------------------------------------------
# io_utils.read_jsonl
# ---------------------------------------------------------------------------
def test_read_jsonl_round_trips_with_write_jsonl(tmp_path):
    from emotion_query_pipeline.io_utils import read_jsonl, write_jsonl
    seg = _seg()
    path = tmp_path / "segments.jsonl"
    write_jsonl(path, [seg])
    records = read_jsonl(path)
    assert records == [seg.model_dump()]


def test_read_jsonl_missing_file_returns_empty(tmp_path):
    from emotion_query_pipeline.io_utils import read_jsonl
    assert read_jsonl(tmp_path / "missing.jsonl") == []


# ---------------------------------------------------------------------------
# TimeChat JSON-array folding (docs/progress_log.md bug fix)
# ---------------------------------------------------------------------------
def test_try_parse_json_array_extracts_full_array():
    raw = ('[{"timestamp": "00:00-00:04", "segment_detail_caption": "a"}, '
           '{"timestamp": "00:05-00:09", "segment_detail_caption": "b"}]')
    scenes = cqt._try_parse_json_array(raw)
    assert scenes == [
        {"timestamp": "00:00-00:04", "segment_detail_caption": "a"},
        {"timestamp": "00:05-00:09", "segment_detail_caption": "b"},
    ]


def test_try_parse_json_array_returns_none_for_object():
    assert cqt._try_parse_json_array('{"a": 1}') is None


def test_try_parse_json_array_returns_none_for_plain_text():
    assert cqt._try_parse_json_array("just a plain caption, no brackets") is None


def test_fold_timechat_scenes_keeps_detail_storyline_acoustics_only():
    scenes = [
        {
            "timestamp": "00:00-00:04",
            "segment_detail_caption": "A woman looks afraid.",
            "storyline": "Fear is established.",
            "acoustics_content": "Tense music.",
            "camera_state": "static shot",
            "video_background": "a basement",
            "speech_content": "",
        },
        {
            "timestamp": "00:05-00:09",
            "segment_detail_caption": "She draws a gun.",
            "storyline": "",
            "acoustics_content": "",
        },
    ]
    folded = cqt._fold_timechat_scenes(scenes)
    assert "(00:00-00:04) A woman looks afraid." in folded
    assert "(Storyline: Fear is established.)" in folded
    assert "(Audio: Tense music.)" in folded
    assert "(00:05-00:09) She draws a gun." in folded
    # Dropped fields never leak into the folded text.
    assert "static shot" not in folded
    assert "a basement" not in folded


def test_fold_timechat_scenes_skips_entries_without_detail_caption():
    folded = cqt._fold_timechat_scenes([{"timestamp": "00:00-00:04", "storyline": "x"}])
    assert folded is None


def test_fold_timechat_scenes_empty_list_returns_none():
    assert cqt._fold_timechat_scenes([]) is None


def test_run_timechat_folds_array_output(monkeypatch):
    # _run_qwen2_5_omni_av is the only heavy call inside _run_timechat; stub it
    # with a fixed multi-scene JSON-array string (shaped like the model's real
    # output) to verify _run_timechat folds it before it ever reaches
    # normalize_to_omni_caption.
    raw_array = (
        '[{"timestamp": "00:00-00:04", "segment_detail_caption": "Establishing shot.", '
        '"storyline": "Sets the scene."}, '
        '{"timestamp": "00:05-00:09", "segment_detail_caption": "She looks afraid.", '
        '"storyline": "Fear begins.", "acoustics_content": "Tense strings."}]'
    )
    monkeypatch.setattr(cqt, "_run_qwen2_5_omni_av", lambda *a, **k: raw_array)
    spec = cqt.get_model_spec("timechat")
    seg = _seg()
    out = cqt._run_timechat(spec, seg, "clip.mp4", cqt.RunnerConfig())
    assert "Establishing shot." in out.raw_output
    assert "She looks afraid." in out.raw_output
    assert "(Storyline: Fear begins.)" in out.raw_output
    assert "(Audio: Tense strings.)" in out.raw_output
    # Folded text now flows through normalize_to_omni_caption as plain AV text.
    cap = cqt.normalize_to_omni_caption(
        out.raw_output, seg, "v", source_caption_model="timechat", modality="av",
    )
    assert cap.caption_status == "normalized"
    assert "She looks afraid." in cap.temporal_description
