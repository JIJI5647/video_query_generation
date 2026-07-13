"""Pure-Python tests for the reference-free caption/emotion-event judge.

No google.genai import, no GPU, no network — sampling, prompt building, verdict
parsing, and aggregation only.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from emotion_query_pipeline import caption_judge as cj

_ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# sample_segments / sample_events
# ---------------------------------------------------------------------------
def _cap(video_id, segment_id):
    return {
        "video_id": video_id, "segment_id": segment_id,
        "visual_description": f"v-{video_id}-{segment_id}",
        "audio_description": f"a-{video_id}-{segment_id}",
        "time_range": [0.0, 5.0],
    }


def test_sample_segments_spreads_across_videos():
    items = (
        [_cap("vA", f"s{i:03d}") for i in range(1, 6)]
        + [_cap("vB", f"s{i:03d}") for i in range(1, 6)]
        + [_cap("vC", f"s{i:03d}") for i in range(1, 6)]
    )
    sample = cj.sample_segments(items, n=6)
    videos = [it["video_id"] for it in sample]
    # 6 requested across 3 videos -> round-robin gives exactly 2 per video.
    assert sorted(videos) == ["vA", "vA", "vB", "vB", "vC", "vC"]


def test_sample_segments_deterministic():
    items = (
        [_cap("vA", f"s{i:03d}") for i in range(1, 6)]
        + [_cap("vB", f"s{i:03d}") for i in range(1, 6)]
    )
    s1 = cj.sample_segments(items, n=4)
    s2 = cj.sample_segments(list(reversed(items)), n=4)
    key = lambda s: [(it["video_id"], it["segment_id"]) for it in s]
    assert key(s1) == key(s2)


def test_sample_segments_n_larger_than_pool_returns_all():
    items = [_cap("vA", "s001"), _cap("vB", "s001")]
    out = cj.sample_segments(items, n=100)
    assert len(out) == 2


def test_sample_segments_n_zero_or_empty():
    assert cj.sample_segments([], n=5) == []
    assert cj.sample_segments([_cap("vA", "s001")], n=0) == []


def test_sample_events_uses_event_id():
    items = [
        {"video_id": "vA", "event_id": "event_2"},
        {"video_id": "vA", "event_id": "event_1"},
        {"video_id": "vB", "event_id": "event_1"},
    ]
    out = cj.sample_events(items, n=2)
    assert len(out) == 2
    assert {it["video_id"] for it in out} == {"vA", "vB"}


# ---------------------------------------------------------------------------
# prompt building
# ---------------------------------------------------------------------------
def test_build_caption_judge_payload_fills_fields():
    prompt = cj.build_caption_judge_payload(_cap("vA", "s001"))
    assert "vA" in prompt and "s001" in prompt
    assert "v-vA-s001" in prompt and "a-vA-s001" in prompt
    assert "{video_id}" not in prompt and "{visual_description}" not in prompt
    assert "faithfulness" in prompt and "hallucination" in prompt


def test_build_event_judge_payload_fills_fields():
    event = {
        "video_id": "vA", "event_id": "event_1", "emotion_label": "happy",
        "event_description": "smiles broadly", "time_range": [10.0, 15.0],
        "visual_evidence": ["smiles broadly"], "audio_evidence": [],
    }
    prompt = cj.build_event_judge_payload(event)
    assert "event_1" in prompt and "happy" in prompt
    assert "smiles broadly" in prompt
    assert "{emotion_label}" not in prompt and "{event_id}" not in prompt
    assert "cue_sufficiency" in prompt and "label_agreement" in prompt


# ---------------------------------------------------------------------------
# parse_caption_verdict
# ---------------------------------------------------------------------------
def _good_caption_raw():
    return {
        "faithfulness": {"reason": "matches", "score": 5},
        "hallucination": {"reason": "nothing invented", "score": 5},
        "coverage": {"reason": "covers most", "score": 4},
        "fluency": {"reason": "clean", "score": 5},
        "emotion_leakage_ok": {"reason": "purely observational", "score": 1},
    }


def test_parse_caption_verdict_good_json():
    v = cj.parse_caption_verdict(_good_caption_raw())
    assert v["parse_error"] is None
    assert v["faithfulness"]["score"] == 5
    assert v["emotion_leakage_ok"]["score"] == 1


def test_parse_caption_verdict_malformed_response_falls_back():
    v = cj.parse_caption_verdict("not even json")
    assert v["parse_error"] is not None
    assert all(v[d]["score"] is None for d in cj.CAPTION_SCORE_DIMENSIONS)


def test_parse_caption_verdict_partial_missing_dimension():
    raw = _good_caption_raw()
    del raw["fluency"]
    v = cj.parse_caption_verdict(raw)
    assert v["fluency"]["score"] is None
    assert "fluency" in v["parse_error"]
    # the rest still parse fine.
    assert v["faithfulness"]["score"] == 5


def test_parse_caption_verdict_out_of_range_score():
    raw = _good_caption_raw()
    raw["coverage"]["score"] = 9
    v = cj.parse_caption_verdict(raw)
    assert v["coverage"]["score"] is None
    assert v["parse_error"] is not None


def test_parse_caption_verdict_emotion_leakage_endpoint_only():
    raw = _good_caption_raw()
    raw["emotion_leakage_ok"]["score"] = 2  # not endpoint 0/1
    v = cj.parse_caption_verdict(raw)
    assert v["emotion_leakage_ok"]["score"] is None


# ---------------------------------------------------------------------------
# parse_event_verdict
# ---------------------------------------------------------------------------
def test_parse_event_verdict_label_agreement_match():
    raw = {
        "cue_sufficiency": {"reason": "sufficient", "score": 4},
        "cue_grounded": {"reason": "grounded", "score": 5},
        "label_agreement": {"reason": "clearly happy", "predicted_label": "happy"},
    }
    v = cj.parse_event_verdict(raw, assigned_label="happy")
    assert v["parse_error"] is None
    assert v["label_agreement"]["score"] == 1
    assert v["label_agreement"]["predicted_label"] == "happy"


def test_parse_event_verdict_label_agreement_mismatch_case_insensitive():
    raw = {
        "cue_sufficiency": {"reason": "ok", "score": 3},
        "cue_grounded": {"reason": "ok", "score": 3},
        "label_agreement": {"reason": "looks sad", "predicted_label": "Sad"},
    }
    v = cj.parse_event_verdict(raw, assigned_label="happy")
    assert v["label_agreement"]["score"] == 0
    assert v["parse_error"] is None  # a valid, just-disagreeing response


def test_parse_event_verdict_malformed_falls_back():
    v = cj.parse_event_verdict({"cue_sufficiency": "garbage"}, assigned_label="happy")
    assert v["parse_error"] is not None
    assert v["cue_sufficiency"]["score"] is None
    assert v["label_agreement"]["score"] is None
    assert v["label_agreement"]["predicted_label"] is None


# ---------------------------------------------------------------------------
# aggregate / combined_caption_score
# ---------------------------------------------------------------------------
def test_aggregate_computes_per_dimension_means():
    verdicts = [
        cj.parse_caption_verdict(_good_caption_raw()),
        cj.parse_caption_verdict({
            "faithfulness": {"reason": "ok", "score": 3},
            "hallucination": {"reason": "ok", "score": 3},
            "coverage": {"reason": "ok", "score": 3},
            "fluency": {"reason": "ok", "score": 3},
            "emotion_leakage_ok": {"reason": "ok", "score": 0},
        }),
    ]
    agg = cj.aggregate(verdicts, cj.CAPTION_SCORE_DIMENSIONS)
    assert agg["n_items"] == 2
    assert agg["n_errors"] == 0
    assert agg["dimensions"]["faithfulness"]["mean"] == 4.0
    assert agg["dimensions"]["faithfulness"]["n"] == 2
    assert agg["dimensions"]["emotion_leakage_ok"]["mean"] == 0.5


def test_aggregate_excludes_none_scores_from_mean():
    verdicts = [
        cj.parse_caption_verdict(_good_caption_raw()),
        cj.parse_caption_verdict("garbage"),  # all None + parse_error set
    ]
    agg = cj.aggregate(verdicts, cj.CAPTION_SCORE_DIMENSIONS)
    assert agg["n_items"] == 2
    assert agg["n_errors"] == 1
    # mean computed only from the one valid verdict, not diluted by the None.
    assert agg["dimensions"]["faithfulness"]["mean"] == 5.0
    assert agg["dimensions"]["faithfulness"]["n"] == 1


def test_combined_caption_score_weighted_mean():
    verdicts = [cj.parse_caption_verdict(_good_caption_raw())]
    agg = cj.aggregate(verdicts, cj.CAPTION_SCORE_DIMENSIONS)
    score = cj.combined_caption_score(agg)
    expected = (5 * 0.30 + 5 * 0.35 + 4 * 0.20 + 5 * 0.15) / 1.0
    assert abs(score - expected) < 1e-9


def test_combined_caption_score_none_when_no_data():
    agg = cj.aggregate([], cj.CAPTION_SCORE_DIMENSIONS)
    assert cj.combined_caption_score(agg) is None


def test_event_aggregate_uses_event_dimensions():
    verdicts = [
        cj.parse_event_verdict(
            {
                "cue_sufficiency": {"reason": "ok", "score": 4},
                "cue_grounded": {"reason": "ok", "score": 5},
                "label_agreement": {"reason": "ok", "predicted_label": "happy"},
            },
            assigned_label="happy",
        ),
    ]
    agg = cj.aggregate(verdicts, cj.EVENT_SCORE_DIMENSIONS)
    assert agg["dimensions"]["cue_sufficiency"]["mean"] == 4.0
    assert agg["dimensions"]["label_agreement"]["mean"] == 1.0


# ---------------------------------------------------------------------------
# import hygiene — this module must not require google.genai
# ---------------------------------------------------------------------------
def test_module_import_has_no_genai_dependency():
    import importlib
    import sys as _sys

    _sys.modules.pop("google.genai", None)
    _sys.modules.pop("google", None)
    importlib.reload(cj)
    assert "google.genai" not in _sys.modules
    assert hasattr(cj, "sample_segments")
