"""Offline unit tests for the Nemotron-3-Nano-Omni caption backend.

Never hits the network or a GPU: ``NemotronOpenAIClient`` is monkeypatched (its
``generate_json`` stubbed to return a canned OmniCaption-shaped dict) so these
exercise only the pure plumbing — prompt building, dispatch, and normalization —
that wires the client's output into the pipeline's ``OmniCaption`` schema.

Run:  python -m pytest tests/test_nemotron_caption.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from emotion_query_pipeline import batch_captioning as bc
from emotion_query_pipeline import caption_query_test as cqt
from emotion_query_pipeline.models import OmniCaption, Segment

# Shaped exactly like the model's real reply per prompts/omni_caption_prompt_unified.txt
# (segment_id/time_range echoed, visual/audio prose, confidence/evidence_strength enums).
_CANNED_REPLY = {
    "segment_id": "s001",
    "time_range": [0.0, 5.0],
    "visual": "A man leans forward across the table, jaw tight, staring at the other person.",
    "audio": "His voice is raised and clipped, with a sharp exhale between words.",
    "confidence": "high",
    "evidence_strength": "clear",
}


def _seg(seg_id="s001", start=0.0, end=5.0, clip="clip.mp4") -> Segment:
    return cqt.make_segment(segment_id=seg_id, start=start, end=end, clip_path=clip)


class _FakeNemotronClient:
    """Stand-in for ``NemotronOpenAIClient`` — records call args, returns canned JSON."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []

    def generate_json(self, prompt, schema_name, video_uri=None):
        self.calls.append((prompt, schema_name, video_uri))
        return dict(_CANNED_REPLY)


def _assert_populated_omni_caption(cap: OmniCaption) -> None:
    assert isinstance(cap, OmniCaption)
    assert cap.segment_id == "s001"
    assert cap.time_range == [0.0, 5.0]
    assert "jaw tight" in cap.visual_description
    assert "raised and clipped" in cap.audio_description
    assert cap.confidence == "high"
    assert cap.evidence_strength == "clear"
    assert cap.caption_status == "normalized"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
def test_nemotron_omni_registered():
    assert "nemotron_omni" in cqt.supported_models()
    spec = cqt.get_model_spec("nemotron_omni")
    assert spec.kind == "av"
    assert spec.requires_video is True
    assert spec.requires_audio is False
    assert spec.default_model_path == "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8"


# ---------------------------------------------------------------------------
# Single-segment runner (caption_query_test._run_nemotron / run_caption_model)
# ---------------------------------------------------------------------------
def test_run_nemotron_builds_prompt_and_normalizes(monkeypatch):
    fake = _FakeNemotronClient()
    monkeypatch.setattr(
        "emotion_query_pipeline.nemotron_client.NemotronOpenAIClient",
        lambda **kwargs: fake,
    )
    seg = _seg()
    config = cqt.RunnerConfig(
        nemotron_base_url="http://0.0.0.0:8000/v1",
        nemotron_model="nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8",
        nemotron_max_tokens=8192,
        nemotron_enable_thinking=True,
    )
    out = cqt.run_caption_model(
        "nemotron_omni", seg, video_path="clip.mp4", audio_path=None, config=config,
    )
    assert out.modality == "av"
    assert isinstance(out.raw_output, dict)
    assert out.source_caption_model == config.nemotron_model

    # The prompt actually reached the fake client and named this clip's segment.
    assert len(fake.calls) == 1
    prompt, schema_name, video_uri = fake.calls[0]
    assert schema_name == "omni_caption"
    assert "s001" in prompt
    assert video_uri == "clip.mp4"

    cap = cqt.normalize_caption_output(out, seg, "vid1", "nemotron_omni")
    _assert_populated_omni_caption(cap)


# ---------------------------------------------------------------------------
# Batch session (batch_captioning.NemotronCaptionSession)
# ---------------------------------------------------------------------------
def test_nemotron_caption_session(monkeypatch):
    fake = _FakeNemotronClient()
    monkeypatch.setattr(
        "emotion_query_pipeline.nemotron_client.NemotronOpenAIClient",
        lambda **kwargs: fake,
    )
    spec = cqt.get_model_spec("nemotron_omni")
    config = cqt.RunnerConfig()
    session = bc.NemotronCaptionSession(spec, config)
    try:
        seg = _seg()
        out = session.caption(seg, video_path="clip.mp4", audio_path=None)
        assert out.modality == "av"
        assert isinstance(out.raw_output, dict)
        cap = cqt.normalize_caption_output(out, seg, "vid1", "nemotron_omni")
        _assert_populated_omni_caption(cap)
    finally:
        session.close()

    assert len(fake.calls) == 1


def test_build_caption_session_dispatches_nemotron(monkeypatch):
    fake = _FakeNemotronClient()
    monkeypatch.setattr(
        "emotion_query_pipeline.nemotron_client.NemotronOpenAIClient",
        lambda **kwargs: fake,
    )
    config = cqt.RunnerConfig()
    session = bc.build_caption_session("nemotron_omni", config)
    try:
        assert isinstance(session, bc.NemotronCaptionSession)
    finally:
        session.close()
