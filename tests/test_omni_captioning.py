"""Local, model-free tests for the Qwen3-Omni captioning backend.

These never import torch/transformers/qwen_omni_utils and never load the
Qwen3-Omni model. They cover: multi-segment prompt construction, robust JSON
array/object extraction, required-field validation, single + multi caption
parsing, the cache/resume decision (segments-per-prompt + prompts-per-call
batching), atomic write, the OmniCaption -> EmotionCaption adapter, the
verify/rewrite LLM client, and that the heavy deps are not imported at module
import time.

Run:  python -m pytest tests/test_omni_captioning.py -q
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from emotion_query_pipeline import omni_captioning as oc
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


# ---------------------------------------------------------------------------
# No heavy imports at module load
# ---------------------------------------------------------------------------
def test_no_heavy_imports_at_module_load():
    for mod in ("torch", "transformers", "qwen_omni_utils"):
        assert mod not in sys.modules, f"{mod} must not be imported at module load"


def test_constructing_captioner_does_not_load_model():
    cap = oc.Qwen3OmniCaptioner()
    assert cap._model is None and cap._processor is None
    assert cap.use_audio_in_video is True
    assert cap.video_reader_backend == "torchvision"
    assert cap.sampling_params == {
        "temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_tokens": 2048,
    }
    for mod in ("torch", "transformers", "qwen_omni_utils"):
        assert mod not in sys.modules


def test_ensure_model_pins_video_reader_backend(monkeypatch):
    monkeypatch.delenv("FORCE_QWENVL_VIDEO_READER", raising=False)
    cap = oc.Qwen3OmniCaptioner(video_reader_backend="torchvision")
    monkeypatch.setattr(cap, "_ensure_model_transformers", lambda: None)
    cap._ensure_model()
    import os as _os
    assert _os.environ["FORCE_QWENVL_VIDEO_READER"] == "torchvision"


# ---------------------------------------------------------------------------
# Prompt construction — N segments, with the clip -> segment mapping
# ---------------------------------------------------------------------------
def test_prompt_lists_every_segment_in_chunk():
    prompt = oc.build_omni_caption_prompt(_segments(3))
    assert "Clip 1 -> s001" in prompt
    assert "Clip 2 -> s002" in prompt
    assert "Clip 3 -> s003" in prompt
    assert "s004" not in prompt


def test_prompt_single_segment_still_lists_it():
    prompt = oc.build_omni_caption_prompt([_segment()])
    assert "Clip 1 -> s022" in prompt and "105.00-110.00s" in prompt


# ---------------------------------------------------------------------------
# JSON extraction (single object + array)
# ---------------------------------------------------------------------------
def test_extract_object_plain_and_fenced():
    assert oc.extract_caption_json(json.dumps(_good_caption_dict()))["segment_id"] == "s022"
    fenced = "ok:\n```json\n" + json.dumps(_good_caption_dict()) + "\n```\n"
    assert oc.extract_caption_json(fenced)["confidence"] == "high"


def test_extract_list_array_and_fenced():
    arr = [_good_caption_dict("s001"), _good_caption_dict("s002")]
    got = oc.extract_caption_list("```json\n" + json.dumps(arr) + "\n```")
    assert [c["segment_id"] for c in got] == ["s001", "s002"]


def test_extract_list_wraps_single_object():
    got = oc.extract_caption_list(json.dumps(_good_caption_dict()))
    assert isinstance(got, list) and len(got) == 1


def test_extract_failures_raise():
    with pytest.raises(oc.CaptionParseError):
        oc.extract_caption_json("no json")
    with pytest.raises(oc.CaptionParseError):
        oc.extract_caption_list("no json")


# ---------------------------------------------------------------------------
# Required-field validation
# ---------------------------------------------------------------------------
def test_missing_required_fields():
    d = _good_caption_dict(); del d["audio_description"]
    assert "audio_description" in oc.missing_required_fields(d)
    d2 = _good_caption_dict(); d2["time_range"] = [1.0]
    assert "time_range" in oc.missing_required_fields(d2)
    # temporal_description is OPTIONAL — its absence must not flag missing.
    d3 = _good_caption_dict(); d3.pop("temporal_description", None)
    assert oc.missing_required_fields(d3) == []
    assert oc.missing_required_fields(_good_caption_dict()) == []


# ---------------------------------------------------------------------------
# parse_caption (single) + parse_captions (multi)
# ---------------------------------------------------------------------------
def test_parse_caption_overrides_metadata():
    d = _good_caption_dict(seg_id="WRONG", time_range=(0.0, 1.0))
    cap = oc.parse_caption(json.dumps(d), _segment(), "vid01")
    assert cap.segment_id == "s022" and cap.time_range == [105.0, 110.0]
    assert cap.video_id == "vid01"


def test_parse_caption_missing_raises_with_raw():
    d = _good_caption_dict(); del d["audio_description"]
    with pytest.raises(oc.CaptionParseError) as ei:
        oc.parse_caption(json.dumps(d), _segment(), "vid01")
    assert ei.value.reason == "missing_required_fields" and ei.value.raw_text


def test_parse_captions_maps_by_segment_id():
    segs = _segments(2)
    arr = [_good_caption_dict("s002"), _good_caption_dict("s001")]  # out of order
    out = oc.parse_captions(json.dumps(arr), segs, "vid01")
    assert set(out) == {"s001", "s002"}
    assert out["s001"].time_range == [5.0, 10.0]  # metadata forced from segment


def test_parse_captions_falls_back_to_position_when_id_unknown():
    segs = _segments(2)
    arr = [_good_caption_dict("zzz"), _good_caption_dict("yyy")]  # bad ids
    out = oc.parse_captions(json.dumps(arr), segs, "vid01")
    assert set(out) == {"s001", "s002"}  # mapped by clip order


def test_parse_captions_skips_invalid_keeps_valid():
    segs = _segments(2)
    bad = _good_caption_dict("s001"); del bad["confidence"]
    arr = [bad, _good_caption_dict("s002")]
    out = oc.parse_captions(json.dumps(arr), segs, "vid01")
    assert set(out) == {"s002"}  # s001 dropped (missing field)


# ---------------------------------------------------------------------------
# Observation-only: captions carry NO emotion field
# ---------------------------------------------------------------------------
def test_caption_has_no_emotion_field():
    cap = OmniCaption.model_validate(_good_caption_dict())
    assert not hasattr(cap, "emotion_description")
    assert cap.temporal_description.startswith("Her voice rises")


# ---------------------------------------------------------------------------
# Atomic write + cache read
# ---------------------------------------------------------------------------
def test_atomic_write_and_read_back(tmp_path):
    path = oc.caption_cache_path(tmp_path, "vid01", "s022")
    oc.atomic_write_json(path, _good_caption_dict())
    assert path.exists() and not path.with_suffix(".json.tmp").exists()
    cap, reason = oc.read_valid_cache(path)
    assert reason is None and cap.segment_id == "s022"


def test_read_valid_cache_states(tmp_path):
    assert oc.read_valid_cache(tmp_path / "nope.json") == (None, "not_found")
    bad = tmp_path / "bad.json"; bad.write_text("{ not json", encoding="utf-8")
    assert oc.read_valid_cache(bad)[1] == "json_parse_error"
    inc = tmp_path / "inc.json"
    d = _good_caption_dict(); del d["visual_objective"]
    inc.write_text(json.dumps(d), encoding="utf-8")
    assert oc.read_valid_cache(inc)[1] == "missing_required_fields"


# ---------------------------------------------------------------------------
# Fake engine (no model) — echoes the segment_ids it sees in the prompt
# ---------------------------------------------------------------------------
class FakeEngine:
    """Implements generate/generate_many; returns a caption array per prompt.

    Reads the `Clip i -> sNNN` mapping out of each prompt and echoes one caption
    object per listed segment, so it behaves like a well-formed model. Records
    the size of every generate_many call (= prompts per call).
    """

    def __init__(self):
        self.call_sizes = []
        self.total_prompts = 0

    @staticmethod
    def _seg_ids(messages):
        text = next(c["text"] for c in messages[0]["content"] if c["type"] == "text")
        return re.findall(r"->\s+(\S+)\s+\(", text)

    def generate_many(self, messages_list):
        self.call_sizes.append(len(messages_list))
        outs = []
        for messages in messages_list:
            self.total_prompts += 1
            arr = [_good_caption_dict(sid) for sid in self._seg_ids(messages)]
            outs.append("```json\n" + json.dumps(arr) + "\n```")
        return outs

    def generate(self, messages):
        return self.generate_many([messages])[0]


def test_generate_then_resume_skips_model(tmp_path):
    cache, raw = tmp_path / "cap", tmp_path / "raw"
    eng, segs = FakeEngine(), _segments(3)
    out = oc.caption_video_omni("v", segs, eng, cache, raw)
    assert len(out) == 3 and eng.total_prompts == 3  # batch 1 -> 3 prompts
    # resume: nothing regenerated
    eng2 = FakeEngine()
    out2 = oc.caption_video_omni("v", segs, eng2, cache, raw)
    assert len(out2) == 3 and eng2.total_prompts == 0


def test_segments_per_prompt_groups_into_one_prompt(tmp_path):
    cache, raw = tmp_path / "cap", tmp_path / "raw"
    eng, segs = FakeEngine(), _segments(4)
    out = oc.caption_video_omni("v", segs, eng, cache, raw, caption_batch_size=2)
    # 4 segments / 2 per prompt = 2 prompts (each generate_many call has 1 prompt)
    assert eng.total_prompts == 2
    assert [c.segment_id for c in out] == ["s001", "s002", "s003", "s004"]


def test_parallel_runs_multiple_prompts_per_call(tmp_path):
    cache, raw = tmp_path / "cap", tmp_path / "raw"
    eng, segs = FakeEngine(), _segments(4)
    oc.caption_video_omni("v", segs, eng, cache, raw,
                          caption_batch_size=1, caption_parallel=2)
    # 4 prompts, 2 per generate call -> calls of size [2, 2]
    assert eng.call_sizes == [2, 2]


def test_overwrite_forces_regen(tmp_path):
    cache, raw = tmp_path / "cap", tmp_path / "raw"
    eng, segs = FakeEngine(), _segments(2)
    oc.caption_video_omni("v", segs, eng, cache, raw)
    assert eng.total_prompts == 2
    oc.caption_video_omni("v", segs, eng, cache, raw, overwrite=True)
    assert eng.total_prompts == 4  # regenerated


def test_invalid_cache_regenerates(tmp_path):
    cache, raw = tmp_path / "cap", tmp_path / "raw"
    p = oc.caption_cache_path(cache, "v", "s001")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{ corrupt", encoding="utf-8")
    eng = FakeEngine()
    oc.caption_video_omni("v", _segments(1), eng, cache, raw)
    assert eng.total_prompts == 1 and oc.read_valid_cache(p)[1] is None


def test_skipped_segment_is_salvaged_and_raw_dumped(tmp_path):
    cache, raw = tmp_path / "cap", tmp_path / "raw"

    class PartialEngine(FakeEngine):
        def generate_many(self, messages_list):
            self.call_sizes.append(len(messages_list))
            outs = []
            for messages in messages_list:
                self.total_prompts += 1
                ids = self._seg_ids(messages)
                # Drop the last segment from the returned array.
                arr = [_good_caption_dict(sid) for sid in ids[:-1]]
                outs.append(json.dumps(arr))
            return outs

    eng = PartialEngine()
    out = oc.caption_video_omni("v", _segments(2), eng, cache, raw,
                                caption_batch_size=2)
    # s002 is now SALVAGED (fed to generation) rather than dropped.
    assert [c.segment_id for c in out] == ["s001", "s002"]
    salvaged = next(c for c in out if c.segment_id == "s002")
    assert salvaged.confidence == "low" and salvaged.evidence_strength == "weak"
    assert getattr(salvaged, "caption_status", None) == "salvaged"
    assert oc.raw_output_path(raw, "v", "s002").exists()  # raw dumped for debug
    # Salvaged captions are NOT cached, so a rerun can regenerate a clean one.
    assert not oc.caption_cache_path(cache, "v", "s002").exists()
    assert oc.caption_cache_path(cache, "v", "s001").exists()


def test_generate_error_does_not_abort(tmp_path):
    cache, raw = tmp_path / "cap", tmp_path / "raw"

    class BoomEngine:
        def generate(self, messages):
            return self.generate_many([messages])[0]

        def generate_many(self, messages_list):
            raise RuntimeError("boom")

    out = oc.caption_video_omni("v", _segments(2), BoomEngine(), cache, raw)
    assert out == []  # nothing produced, but no exception escaped


# ---------------------------------------------------------------------------
# QwenOmniLLMClient (verify/rewrite)
# ---------------------------------------------------------------------------
class ReplyEngine:
    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []

    def generate(self, messages):
        self.seen.append(messages)
        return self.replies.pop(0)

    def generate_many(self, messages_list):
        return [self.generate(m) for m in messages_list]


def test_build_messages_multi():
    msgs = oc.Qwen3OmniCaptioner._build_messages_multi("judge", ["a.mp4", "b.mp4"])
    content = msgs[0]["content"]
    assert [c["video"] for c in content if c["type"] == "video"] == ["a.mp4", "b.mp4"]
    assert content[-1]["text"] == "judge"


def test_llm_client_parses_and_passes_clips():
    payload = {"video_id": "v", "round_index": 1, "results": []}
    eng = ReplyEngine(["```json\n" + json.dumps(payload) + "\n```"])
    out = oc.QwenOmniLLMClient(eng).generate_json("p", "VerificationBatchOutput",
                                                  video_uri=["c1.mp4"])
    assert out == payload
    vids = [c for c in eng.seen[0][0]["content"] if c["type"] == "video"]
    assert [v["video"] for v in vids] == ["c1.mp4"]


def test_llm_client_retries_then_raises():
    eng = ReplyEngine(["no json", "still none"])
    with pytest.raises(RuntimeError):
        oc.QwenOmniLLMClient(eng, max_retries=2).generate_json(
            "p", "VerificationBatchOutput", video_uri="c.mp4"
        )
    assert not eng.replies


def test_llm_client_usage_report_shape():
    rep = oc.QwenOmniLLMClient(ReplyEngine([])).usage_report()
    assert rep["total"]["total_tokens"] == 0 and rep["by_stage"] == {}
