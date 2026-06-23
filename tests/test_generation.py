"""Model-free tests for feeding rich OmniCaption (and flat EmotionCaption) to
the query-generation stage. No SDK / network: a fake LLM client is injected and
``google.genai`` is never imported (it is now lazy in llm_client).

Run:  python -m pytest tests/test_generation.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from emotion_query_pipeline import generation as gen
from emotion_query_pipeline.models import EmotionCaption, OmniCaption, Segment


def _segments():
    return [
        Segment(segment_id="s001", index=1, start_time=0.0, end_time=5.0, clip_path="c1.mp4"),
        Segment(segment_id="s002", index=2, start_time=5.0, end_time=10.0, clip_path="c2.mp4"),
    ]


def _omni(seg_id, start, end):
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
        "audio_description": "The woman speaks with an excited voice.",
        "emotion_description": "The woman appears surprised, supported by widened eyes.",
        "confidence": "high", "evidence_strength": "clear",
    })


def _flat(seg_id):
    return EmotionCaption(
        video_id="v", caption_id=f"v_{seg_id}", segment_ids=[seg_id],
        person="a man in a suit", action="raising his voice", sound="shouting",
        emotion="angry", confidence="high", evidence_strength="clear",
        observable_evidence=["furrowed brow"],
    )


class FakeClient:
    """BaseLLMClient-compatible: returns canned queries, records the prompt."""

    def __init__(self, queries):
        self.queries = queries
        self.last_prompt = None

    def generate_json(self, prompt, schema_name, video_uri=None):
        self.last_prompt = prompt
        return {"queries": list(self.queries)}


def test_omni_payload_keeps_rich_structure():
    seg_time = gen._segment_time_map(_segments())
    payload = gen._captions_payload([_omni("s001", 0.0, 5.0)], seg_time)
    e = payload[0]
    assert e["time_range"] == [0.0, 5.0]
    # Full nested structure preserved (not flattened to person/action strings).
    assert e["visual_objective"]["people"][0]["person"] == "a woman in a white shirt"
    assert e["visual_expression"][0]["facial_cues"] == ["eyes widened"]
    # Full sentences, not a compressed label.
    assert e["audio_description"] == "The woman speaks with an excited voice."
    assert "surprised" in e["emotion_description"] and len(e["emotion_description"]) > 20


def test_flat_caption_maps_into_same_schema():
    seg_time = gen._segment_time_map(_segments())
    e = gen._captions_payload([_flat("s001")], seg_time)[0]
    # Same keys as the omni payload, filled sparsely from the flat caption.
    assert e["visual_objective"]["people"][0]["person"] == "a man in a suit"
    assert e["visual_expression"][0]["facial_cues"] == ["furrowed brow"]
    assert e["audio_description"] == "shouting"
    assert e["emotion_description"] == "angry"


def test_build_prompt_includes_rich_fields():
    prompt = gen.build_generation_prompt("v", [_omni("s001", 0.0, 5.0)], _segments())
    assert "visual_objective" in prompt and "visual_expression" in prompt
    assert "excited voice" in prompt


def test_generate_queries_resolves_segment_ids_and_provenance():
    caps = [_omni("s001", 0.0, 5.0), _omni("s002", 5.0, 10.0)]
    client = FakeClient([
        {"query_type": "emotion_state",
         "query_text": "When does the woman appear surprised?",
         "time_range": [1.0, 4.0]},
    ])
    out = gen.generate_queries("v", caps, client, _segments())
    assert len(out.queries) == 1
    q = out.queries[0]
    assert q.segment_ids == ["s001"]  # resolved from time_range
    assert q.source_caption_ids == ["v_s001"]  # caption id synthesized from segment


def test_generate_queries_drops_out_of_range_query():
    caps = [_omni("s001", 0.0, 5.0)]
    client = FakeClient([
        {"query_type": "evidence_cue", "query_text": "x", "time_range": [100.0, 110.0]},
    ])
    out = gen.generate_queries("v", caps, client, _segments())
    assert out.queries == []  # range outside the video -> dropped
