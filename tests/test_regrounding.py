"""Model-free tests for the re-grounding stage (generation -> re-ground ->
verify). No SDK / network: a fake LLM client is injected, exactly like
tests/test_generation.py.

Run:  python -m pytest tests/test_regrounding.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from emotion_query_pipeline import regrounding as rg
from emotion_query_pipeline.models import EventGroundedQuery, OmniCaption, Segment


def _segments(n=5):
    return [
        Segment(
            segment_id=f"s{str(i).zfill(3)}", index=i,
            start_time=(i - 1) * 5.0, end_time=i * 5.0, clip_path=f"c{i}.mp4",
        )
        for i in range(1, n + 1)
    ]


def _omni(seg_id, start, end, text="something happens"):
    return OmniCaption.model_validate({
        "segment_id": seg_id,
        "time_range": [start, end],
        "visual_description": text,
        "audio_description": "",
        "confidence": "high", "evidence_strength": "clear",
    })


def _captions(segments):
    return [_omni(s.segment_id, s.start_time, s.end_time, f"evidence at {s.segment_id}") for s in segments]


def _query(qid="v_q01", seg_ids=("s002",), text="When does she look surprised?"):
    segs = _segments()
    seg_time = {s.segment_id: (s.start_time, s.end_time) for s in segs}
    spans = [seg_time[sid] for sid in seg_ids]
    tr = [min(s for s, _ in spans), max(e for _, e in spans)]
    return EventGroundedQuery(
        video_id="v", query_id=qid, query_type="emotion_state", query_text=text,
        time_range=tr, segment_ids=list(seg_ids), source_caption_ids=[f"v_{seg_ids[0]}"],
    )


class FakeClient:
    """BaseLLMClient-compatible: returns a canned groundings mapping."""

    def __init__(self, groundings):
        self.groundings = groundings
        self.last_prompt = None
        self.calls = 0

    def generate_json(self, prompt, schema_name, video_uri=None):
        self.calls += 1
        self.last_prompt = prompt
        assert schema_name == "RegroundingOutput"
        assert video_uri is None  # text-only, no video
        return {"video_id": "v", "groundings": list(self.groundings)}


class FailingClient:
    def generate_json(self, prompt, schema_name, video_uri=None):
        raise RuntimeError("simulated API failure")


def test_full_scope_updates_grounding_and_preserves_original():
    segments = _segments()
    captions = _captions(segments)
    q = _query(seg_ids=("s002",))
    client = FakeClient([{"query_id": "v_q01", "segment_ids": ["s004"]}])

    updated, stats = rg.reground_queries(
        "v", [q], captions, segments, client, scope="full",
    )
    assert len(updated) == 1
    nq = updated[0]
    # Final grounding is Gemini's pick.
    assert nq.segment_ids == ["s004"]
    assert nq.time_range == [15.0, 20.0]
    # Original generation-stage grounding is preserved, not lost.
    assert nq.gen_segment_ids == ["s002"]
    assert nq.gen_time_range == [5.0, 10.0]
    assert stats == {"total": 1, "changed": 1, "fallback": 0}
    # ONE call for the whole video.
    assert client.calls == 1


def test_full_scope_prompt_contains_all_segments_for_every_query():
    segments = _segments()
    captions = _captions(segments)
    q = _query(seg_ids=("s002",))
    client = FakeClient([{"query_id": "v_q01", "segment_ids": ["s002"]}])
    rg.reground_queries("v", [q], captions, segments, client, scope="full")

    prompt_payload = rg.build_regrounding_prompt(
        "v", [q], captions, segments, scope="full",
    )
    candidates = json.loads(
        prompt_payload.split("QUERIES (each with its own candidate segments):\n", 1)[1]
        .split("\n\nOUTPUT REQUIREMENTS", 1)[0]
    )
    assert len(candidates) == 1
    seg_ids_seen = [c["segment_id"] for c in candidates[0]["candidate_segments"]]
    assert seg_ids_seen == [s.segment_id for s in segments]  # ALL segments, in order


def test_window_scope_limits_candidates_to_window():
    segments = _segments(n=10)
    captions = _captions(segments)
    q = _query(seg_ids=("s005",))  # index 5
    prompt_payload = rg.build_regrounding_prompt(
        "v", [q], captions, segments, scope="window", window=2,
    )
    candidates = json.loads(
        prompt_payload.split("QUERIES (each with its own candidate segments):\n", 1)[1]
        .split("\n\nOUTPUT REQUIREMENTS", 1)[0]
    )
    seg_ids_seen = [c["segment_id"] for c in candidates[0]["candidate_segments"]]
    # +/- 2 segments around s005 (index 5) -> s003..s007.
    assert seg_ids_seen == ["s003", "s004", "s005", "s006", "s007"]

    # A pick outside the window is impossible to satisfy: the client returns a
    # segment not in the candidate list, and it must fall back.
    client = FakeClient([{"query_id": "v_q01", "segment_ids": ["s009"]}])
    updated, stats = rg.reground_queries(
        "v", [q], captions, segments, client, scope="window", window=2,
    )
    assert updated[0].segment_ids == ["s005"]  # fell back to original
    assert stats["fallback"] == 1


def test_invalid_selection_falls_back_others_still_update():
    segments = _segments()
    captions = _captions(segments)
    q1 = _query(qid="v_q01", seg_ids=("s001",))
    q2 = _query(qid="v_q02", seg_ids=("s002",))
    client = FakeClient([
        {"query_id": "v_q01", "segment_ids": []},  # empty -> invalid
        {"query_id": "v_q02", "segment_ids": ["s003"]},  # valid, different segment
    ])
    updated, stats = rg.reground_queries(
        "v", [q1, q2], captions, segments, client, scope="full",
    )
    by_id = {u.query_id: u for u in updated}
    # Fell back: kept original grounding, gen_* still records it.
    assert by_id["v_q01"].segment_ids == ["s001"]
    assert by_id["v_q01"].gen_segment_ids == ["s001"]
    # Updated normally.
    assert by_id["v_q02"].segment_ids == ["s003"]
    assert by_id["v_q02"].gen_segment_ids == ["s002"]
    assert stats == {"total": 2, "changed": 1, "fallback": 1}


def test_non_contiguous_selection_falls_back():
    segments = _segments()
    captions = _captions(segments)
    q = _query(seg_ids=("s002",))
    client = FakeClient([{"query_id": "v_q01", "segment_ids": ["s001", "s003"]}])
    updated, stats = rg.reground_queries("v", [q], captions, segments, client, scope="full")
    assert updated[0].segment_ids == ["s002"]
    assert stats["fallback"] == 1


def test_missing_query_in_response_falls_back():
    segments = _segments()
    captions = _captions(segments)
    q = _query(seg_ids=("s002",))
    client = FakeClient([])  # no groundings at all
    updated, stats = rg.reground_queries("v", [q], captions, segments, client, scope="full")
    assert updated[0].segment_ids == ["s002"]
    assert updated[0].gen_segment_ids == ["s002"]
    assert stats["fallback"] == 1


def test_client_call_failure_falls_back_for_all():
    segments = _segments()
    captions = _captions(segments)
    q1 = _query(qid="v_q01", seg_ids=("s001",))
    q2 = _query(qid="v_q02", seg_ids=("s002",))
    updated, stats = rg.reground_queries(
        "v", [q1, q2], captions, segments, FailingClient(), scope="full",
    )
    assert [u.segment_ids for u in updated] == [["s001"], ["s002"]]
    assert stats == {"total": 2, "changed": 0, "fallback": 2}


def test_no_queries_is_a_noop():
    segments = _segments()
    captions = _captions(segments)
    updated, stats = rg.reground_queries("v", [], captions, segments, FakeClient([]))
    assert updated == []
    assert stats == {"total": 0, "changed": 0, "fallback": 0}


def test_disabled_regrounding_path_leaves_queries_unchanged():
    """Mirrors how run_pipeline.py/rerun_generation.py gate the call: when
    --no-regrounding is passed, reground_queries is simply never invoked, so
    gen_* stays at its EventGroundedQuery default (empty) and time_range/
    segment_ids are exactly what generation produced.
    """
    q = _query(seg_ids=("s002",))
    assert q.gen_time_range == []
    assert q.gen_segment_ids == []
    assert q.segment_ids == ["s002"]
    assert q.time_range == [5.0, 10.0]
