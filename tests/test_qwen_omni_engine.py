"""Local, model-free tests for the Qwen3-Omni verify/rewrite engine + client.

Never import torch/transformers/qwen_omni_utils and never load the model.

Run:  python -m pytest tests/test_qwen_omni_engine.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from emotion_query_pipeline import qwen_omni_engine as qe


# ---------------------------------------------------------------------------
# Lazy load: nothing heavy at import / construct time
# ---------------------------------------------------------------------------
def test_no_heavy_imports_at_module_load():
    for mod in ("torch", "transformers", "qwen_omni_utils"):
        assert mod not in sys.modules, f"{mod} must not be imported at module load"


def test_constructing_engine_does_not_load_model():
    eng = qe.Qwen3OmniCaptioner()
    assert eng._model is None and eng._processor is None
    assert eng.use_audio_in_video is True
    assert eng.video_reader_backend == "torchvision"
    for mod in ("torch", "transformers", "qwen_omni_utils"):
        assert mod not in sys.modules


def test_ensure_model_pins_video_reader_backend(monkeypatch):
    monkeypatch.delenv("FORCE_QWENVL_VIDEO_READER", raising=False)
    eng = qe.Qwen3OmniCaptioner(video_reader_backend="torchvision")
    monkeypatch.setattr(eng, "_ensure_model_transformers", lambda: None)
    eng._ensure_model()
    import os as _os
    assert _os.environ["FORCE_QWENVL_VIDEO_READER"] == "torchvision"


def test_build_messages_multi():
    msgs = qe.Qwen3OmniCaptioner._build_messages_multi("judge", ["a.mp4", "b.mp4"])
    content = msgs[0]["content"]
    assert [c["video"] for c in content if c["type"] == "video"] == ["a.mp4", "b.mp4"]
    assert content[-1]["text"] == "judge"


# ---------------------------------------------------------------------------
# Verify/rewrite client (fake engine, no model)
# ---------------------------------------------------------------------------
class ReplyEngine:
    """Returns canned raw strings; records the messages it was given."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []

    def generate(self, messages):
        self.seen.append(messages)
        return self.replies.pop(0)

    def generate_many(self, messages_list):
        return [self.generate(m) for m in messages_list]


def test_llm_client_parses_and_passes_clips():
    payload = {"video_id": "v", "round_index": 1, "results": []}
    eng = ReplyEngine(["```json\n" + json.dumps(payload) + "\n```"])
    out = qe.QwenOmniLLMClient(eng).generate_json(
        "p", "VerificationBatchOutput", video_uri=["c1.mp4"]
    )
    assert out == payload
    vids = [c for c in eng.seen[0][0]["content"] if c["type"] == "video"]
    assert [v["video"] for v in vids] == ["c1.mp4"]


def test_llm_client_retries_then_raises():
    eng = ReplyEngine(["no json", "still none"])
    with pytest.raises(RuntimeError):
        qe.QwenOmniLLMClient(eng, max_retries=2).generate_json(
            "p", "VerificationBatchOutput", video_uri="c.mp4"
        )
    assert not eng.replies


def test_llm_client_many_batches_and_falls_back():
    # Two prompts: first parses, second is garbage -> per-item fallback retries it.
    good = json.dumps({"results": []})
    eng = ReplyEngine([good, "garbage", good])  # batch:[good,garbage]; fallback:good
    out = qe.QwenOmniLLMClient(eng).generate_json_many(
        ["p1", "p2"], "VerificationBatchOutput", video_uris=["a.mp4", "b.mp4"]
    )
    assert out == [{"results": []}, {"results": []}]


def test_llm_client_usage_report_shape():
    rep = qe.QwenOmniLLMClient(ReplyEngine([])).usage_report()
    assert rep["total"]["total_tokens"] == 0 and rep["by_stage"] == {}
