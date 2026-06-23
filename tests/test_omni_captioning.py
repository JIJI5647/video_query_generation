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
def test_ensure_model_pins_video_reader_backend(monkeypatch):
    monkeypatch.delenv("FORCE_QWENVL_VIDEO_READER", raising=False)
    cap = oc.Qwen3OmniCaptioner(engine="vllm", video_reader_backend="torchvision")
    # Stub the heavy loader so no model/vllm import happens.
    monkeypatch.setattr(cap, "_ensure_model_vllm", lambda: None)
    cap._ensure_model()
    import os as _os
    assert _os.environ["FORCE_QWENVL_VIDEO_READER"] == "torchvision"


def test_no_heavy_imports_at_module_load():
    for mod in ("vllm", "torch", "transformers", "qwen_omni_utils"):
        assert mod not in sys.modules, f"{mod} must not be imported at module load"


def test_constructing_captioner_does_not_load_model():
    cap = oc.Qwen3OmniCaptioner()  # cheap: stores config only
    assert cap._llm is None and cap._processor is None
    assert cap.use_audio_in_video is True
    assert cap.video_reader_backend == "torchvision"  # avoid torchcodec by default
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


class BatchCaptioner:
    """Implements caption_many; records the size of each batched call."""

    def __init__(self, payload: dict):
        self.payload = payload
        self.batch_calls = []  # one entry per caption_many call = its size
        self.single_calls = 0

    def _raw(self, seg_id):
        d = dict(self.payload)
        d["segment_id"] = seg_id  # echo (parse_caption overrides anyway)
        return "```json\n" + json.dumps(d) + "\n```"

    def caption(self, prompt_text, clip_path):
        self.single_calls += 1
        return self._raw("sX")

    def caption_many(self, items):
        self.batch_calls.append(len(items))
        return [self._raw(f"s{i}") for i, _ in enumerate(items)]


def _segments(n):
    return [
        Segment(segment_id=f"s{i:03d}", index=i, start_time=i * 5.0,
                end_time=i * 5.0 + 5.0, clip_path=f"c{i}.mp4")
        for i in range(1, n + 1)
    ]


def test_batched_groups_prompts_and_preserves_order(tmp_path):
    cache, raw = tmp_path / "captions", tmp_path / "raw"
    fake = BatchCaptioner(_good_caption_dict())
    segs = _segments(5)
    out = oc.caption_video_omni(
        "vid01", segs, fake, cache, raw, caption_batch_size=2
    )
    # 5 segments / batch 2 -> calls of size [2, 2, 1], no per-segment fallback.
    assert fake.batch_calls == [2, 2, 1]
    assert fake.single_calls == 0
    assert len(out) == 5
    # Metadata is forced from each segment, in original order.
    assert [c.segment_id for c in out] == [s.segment_id for s in segs]


def test_batched_resume_skips_cached_and_only_generates_misses(tmp_path):
    cache, raw = tmp_path / "captions", tmp_path / "raw"
    fake = BatchCaptioner(_good_caption_dict())
    segs = _segments(4)
    # First pass generates all 4 (batch 2 -> [2, 2]).
    oc.caption_video_omni("vid01", segs, fake, cache, raw, caption_batch_size=2)
    assert fake.batch_calls == [2, 2]
    # Second pass: all cached -> no model calls at all.
    fake.batch_calls.clear()
    out = oc.caption_video_omni("vid01", segs, fake, cache, raw, caption_batch_size=2)
    assert fake.batch_calls == [] and fake.single_calls == 0
    assert len(out) == 4


def test_batched_falls_back_to_per_segment_on_error(tmp_path):
    cache, raw = tmp_path / "captions", tmp_path / "raw"

    class FlakyBatch(BatchCaptioner):
        def caption_many(self, items):
            raise RuntimeError("boom")  # whole batch fails

    fake = FlakyBatch(_good_caption_dict())
    segs = _segments(3)
    out = oc.caption_video_omni("vid01", segs, fake, cache, raw, caption_batch_size=3)
    # Degraded to per-segment caption() for all 3.
    assert fake.single_calls == 3
    assert len(out) == 3


def test_parse_failure_saves_raw_and_skips(tmp_path):
    cache, raw = tmp_path / "captions", tmp_path / "raw"

    class BadCaptioner:
        def caption(self, prompt_text, clip_path):
            return "the model rambled and produced no json"

    out = oc.caption_video_omni("vid01", [_segment()], BadCaptioner(), cache, raw)
    assert out == []  # segment skipped, video not aborted
    assert oc.raw_output_path(raw, "vid01", "s022").exists()  # raw saved for debug


# ---------------------------------------------------------------------------
# QwenOmniLLMClient (verify/rewrite) — model-free via a fake engine
# ---------------------------------------------------------------------------
class FakeEngine:
    """Stand-in for Qwen3OmniCaptioner: records messages, returns canned text."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.seen_messages = []

    def generate(self, messages):
        self.seen_messages.append(messages)
        return self.replies.pop(0)


def test_build_messages_multi_has_one_video_per_clip():
    msgs = oc.Qwen3OmniCaptioner._build_messages_multi(
        "judge this", ["a.mp4", "b.mp4"]
    )
    content = msgs[0]["content"]
    videos = [c for c in content if c["type"] == "video"]
    texts = [c for c in content if c["type"] == "text"]
    assert [v["video"] for v in videos] == ["a.mp4", "b.mp4"]
    assert texts[-1]["text"] == "judge this"


def test_build_messages_multi_text_only_when_no_clips():
    msgs = oc.Qwen3OmniCaptioner._build_messages_multi("text only", [])
    content = msgs[0]["content"]
    assert all(c["type"] == "text" for c in content)


def test_llm_client_parses_json_and_passes_clips():
    payload = {"video_id": "v", "round_index": 1, "results": []}
    engine = FakeEngine(["```json\n" + json.dumps(payload) + "\n```"])
    client = oc.QwenOmniLLMClient(engine)
    out = client.generate_json("p", "VerificationBatchOutput", video_uri=["c1.mp4"])
    assert out == payload
    # The single clip path was turned into one video part (no upload).
    videos = [c for c in engine.seen_messages[0][0]["content"] if c["type"] == "video"]
    assert [v["video"] for v in videos] == ["c1.mp4"]


def test_llm_client_retries_then_raises():
    engine = FakeEngine(["no json", "still no json"])
    client = oc.QwenOmniLLMClient(engine, max_retries=2)
    with pytest.raises(RuntimeError):
        client.generate_json("p", "VerificationBatchOutput", video_uri="c.mp4")
    assert not engine.replies  # both attempts consumed


def test_llm_client_usage_report_shape():
    rep = oc.QwenOmniLLMClient(FakeEngine([])).usage_report()
    assert rep["total"]["total_tokens"] == 0 and rep["by_stage"] == {}
