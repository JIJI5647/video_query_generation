"""Model-free tests for feeding observation OmniCaptions (+ emotion events) to
the query-generation stage. No SDK / network: a fake LLM client is injected and
``google.genai`` is never imported (it is lazy in llm_client).

Run:  python -m pytest tests/test_generation.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from emotion_query_pipeline import generation as gen
from emotion_query_pipeline.models import (
    EmotionCaption,
    EmotionEvent,
    OmniCaption,
    Segment,
)


def _segments():
    return [
        Segment(segment_id="s001", index=1, start_time=0.0, end_time=5.0, clip_path="c1.mp4"),
        Segment(segment_id="s002", index=2, start_time=5.0, end_time=10.0, clip_path="c2.mp4"),
    ]


def _omni(seg_id, start, end):
    return OmniCaption.model_validate({
        "segment_id": seg_id,
        "time_range": [start, end],
        "visual_description": "A woman in a white shirt talks, eyes widened, gaze "
                               "toward the man.",
        "audio_description": "The woman speaks with a raised voice.",
        "confidence": "high", "evidence_strength": "clear",
    })


def _legacy_structured_omni(seg_id, start, end):
    """An old-format cached OmniCaption: structured fields, no visual_description."""
    return OmniCaption.model_validate({
        "segment_id": seg_id,
        "time_range": [start, end],
        "visual_objective": {
            "people": [{"person": "a woman in a white shirt", "action": "talking"}],
            "scene": {"location": "indoor room", "setting": "conversation"},
            "objects": [], "interactions": [], "key_actions": [], "visibility_notes": "",
        },
        "visual_expression": [{
            "person": "the woman in a white shirt",
            "facial_cues": ["eyes widened"], "body_cues": [], "gaze": "toward the man",
        }],
        "audio_description": "The woman speaks with a raised voice.",
        "temporal_description": "Her voice rises sharply midway through the clip.",
        "confidence": "high", "evidence_strength": "clear",
    })


def _flat(seg_id):
    return EmotionCaption(
        video_id="v", caption_id=f"v_{seg_id}", segment_ids=[seg_id],
        person="a man in a suit", action="raising his voice", sound="shouting",
        emotion="angry", confidence="high", evidence_strength="clear",
        observable_evidence=["furrowed brow"],
    )


def _event(start, end):
    return EmotionEvent(
        video_id="v", event_id="v_e01", emotion_label="surprised",
        event_description="the woman widens her eyes",
        time_range=[start, end], target_person_or_group="the woman in a white shirt",
        visual_evidence=["eyes widened"], audio_evidence=["raised voice"],
        confidence="high", evidence_strength="clear",
    )


class FakeClient:
    """BaseLLMClient-compatible: returns canned queries, records the prompt."""

    def __init__(self, queries):
        self.queries = queries
        self.last_prompt = None

    def generate_json(self, prompt, schema_name, video_uri=None):
        self.last_prompt = prompt
        return {"queries": list(self.queries)}


def test_omni_payload_concatenates_visual_and_audio():
    seg_time = gen._segment_time_map(_segments())
    payload = gen._captions_payload([_omni("s001", 0.0, 5.0)], seg_time)
    e = payload[0]
    assert e["time_range"] == [0.0, 5.0]
    assert e["caption"] == (
        "Visual: A woman in a white shirt talks, eyes widened, gaze toward the man.\n"
        "Audio: The woman speaks with a raised voice."
    )
    # Observation-only: NO emotion field anywhere in the payload, and no more
    # structured sub-fields (superseded by the single "caption" string).
    assert "emotion_description" not in e and "emotion" not in e
    assert "visual_objective" not in e and "visual_expression" not in e


def test_flat_caption_maps_into_same_schema():
    seg_time = gen._segment_time_map(_segments())
    e = gen._captions_payload([_flat("s001")], seg_time)[0]
    # Legacy flat EmotionCaption is synthesized into the same "caption" prose.
    assert "a man in a suit" in e["caption"]
    assert "furrowed brow" in e["caption"]
    assert "Audio: shouting" in e["caption"]
    assert "emotion_description" not in e and "emotion" not in e


def test_payload_entry_falls_back_for_legacy_structured_caption():
    """A legacy cached OmniCaption (no visual_description) still serializes."""
    seg_time = gen._segment_time_map(_segments())
    e = gen._captions_payload([_legacy_structured_omni("s001", 0.0, 5.0)], seg_time)[0]
    assert e["caption"].startswith("Visual: ")
    assert "a woman in a white shirt" in e["caption"]
    assert "talking" in e["caption"]
    assert "eyes widened" in e["caption"]
    assert "Audio: The woman speaks with a raised voice." in e["caption"]


def test_payload_entry_omits_audio_line_when_empty():
    seg_time = gen._segment_time_map(_segments())
    caption = OmniCaption.model_validate({
        "segment_id": "s001", "time_range": [0.0, 5.0],
        "visual_description": "A man in a suit paces the room.",
        "audio_description": "", "confidence": "medium", "evidence_strength": "weak",
    })
    e = gen._captions_payload([caption], seg_time)[0]
    assert e["caption"] == "Visual: A man in a suit paces the room."
    assert "Audio:" not in e["caption"]


def test_build_prompt_includes_caption_and_event_fields():
    prompt = gen.build_generation_prompt(
        "v", [_omni("s001", 0.0, 5.0)], [_event(0.0, 5.0)], _segments()
    )
    assert "Visual: A woman in a white shirt" in prompt
    assert "raised voice" in prompt
    # The emotion signal comes from the events payload.
    assert "surprised" in prompt


def test_generate_queries_resolves_segment_ids_and_provenance():
    caps = [_omni("s001", 0.0, 5.0), _omni("s002", 5.0, 10.0)]
    client = FakeClient([
        {"query_type": "emotion_state",
         "query_text": "When does the woman appear surprised?",
         "time_range": [1.0, 4.0]},
    ])
    out = gen.generate_queries("v", caps, [_event(1.0, 4.0)], client, _segments())
    assert len(out.queries) == 1
    q = out.queries[0]
    assert q.segment_ids == ["s001"]  # resolved from time_range
    assert q.source_caption_ids == ["v_s001"]  # caption id synthesized from segment


def test_generate_queries_drops_out_of_range_query():
    caps = [_omni("s001", 0.0, 5.0)]
    client = FakeClient([
        {"query_type": "evidence_cue", "query_text": "x", "time_range": [100.0, 110.0]},
    ])
    out = gen.generate_queries("v", caps, [_event(0.0, 5.0)], client, _segments())
    assert out.queries == []  # range outside the video -> dropped


def test_generate_queries_no_events_returns_empty():
    caps = [_omni("s001", 0.0, 5.0)]
    client = FakeClient([
        {"query_type": "emotion_state", "query_text": "x", "time_range": [1.0, 4.0]},
    ])
    out = gen.generate_queries("v", caps, [], client, _segments())
    assert out.queries == []  # no emotion events -> nothing to ground on
