"""Emotion-event stage: runaway-generation guard.

Gemini occasionally emits FAR more events than captions on long/verbose inputs
(observed 57 captions -> 180 events in one draw; a clean re-draw gave 17). The
guard in ``generate_emotion_events`` re-samples such draws and keeps the
least-inflated attempt, truncating to a ceiling if every attempt is runaway.
These tests inject a fake client (no Gemini) to exercise that logic.
"""
from emotion_query_pipeline.emotion_events import (
    _DEGENERATE_EVENT_MULTIPLE,
    generate_emotion_events,
    merge_contiguous_events,
)
from emotion_query_pipeline.models import EmotionEvent, OmniCaption, Segment


def _ev(event_id, label, start, end, target="the man in the blue jacket", **kw):
    fields = dict(
        video_id="v", event_id=event_id, emotion_label=label,
        event_description=kw.pop("event_description", "d"),
        time_range=[start, end], target_person_or_group=target,
        visual_evidence=kw.pop("visual_evidence", []),
        audio_evidence=kw.pop("audio_evidence", []),
        confidence=kw.pop("confidence", "medium"),
        evidence_strength=kw.pop("evidence_strength", "ambiguous"),
    )
    fields.update(kw)
    return EmotionEvent(**fields)


def _caps_segs(n):
    caps = [
        OmniCaption(
            video_id="v", segment_id=f"s{i:03d}",
            time_range=[i * 5.0, (i + 1) * 5.0], temporal_description="x",
        )
        for i in range(n)
    ]
    segs = [
        Segment(
            segment_id=f"s{i:03d}", index=i,
            start_time=i * 5.0, end_time=(i + 1) * 5.0, clip_path=f"/c/s{i}.mp4",
        )
        for i in range(n)
    ]
    return caps, segs


class _FakeClient:
    """Returns a preset event-count per successive call (last value repeats)."""

    def __init__(self, counts):
        self.counts = list(counts)
        self.calls = 0

    def generate_json(self, prompt, schema_name, video_uri=None):
        n = self.counts[min(self.calls, len(self.counts) - 1)]
        self.calls += 1
        return {
            "video_id": "v",
            "events": [
                # Distinct target_person_or_group per event so the new
                # merge-contiguous-events post-processing (Change 3) never
                # collapses these — this fixture is only exercising the
                # runaway-generation guard's event COUNTING logic.
                {
                    "event_id": f"e{i}", "emotion_label": "fear",
                    "event_description": "d", "time_range": [0.0, 5.0],
                    "target_person_or_group": f"p{i}",
                }
                for i in range(n)
            ],
        }


def test_runaway_draw_is_resampled_and_good_draw_kept():
    caps, segs = _caps_segs(57)
    client = _FakeClient([180, 17])  # runaway, then healthy
    out = generate_emotion_events("v", caps, client, segs)
    assert client.calls == 2          # re-sampled once
    assert len(out.events) == 17      # kept the healthy draw


def test_all_runaway_draws_are_truncated_to_ceiling():
    caps, segs = _caps_segs(57)
    client = _FakeClient([180, 180, 180])  # never recovers
    out = generate_emotion_events("v", caps, client, segs)
    assert client.calls == 3                             # exhausts retries
    ceiling = _DEGENERATE_EVENT_MULTIPLE * 57
    assert len(out.events) == ceiling                    # truncated, not 180


def test_healthy_first_draw_is_not_resampled():
    caps, segs = _caps_segs(57)
    client = _FakeClient([8])
    out = generate_emotion_events("v", caps, client, segs)
    assert client.calls == 1          # no wasted re-sample
    assert len(out.events) == 8


def test_short_video_below_floor_is_not_flagged():
    # 3 captions: ceiling = max(2*3, floor=10) = 10, so a plausible 6-event draw
    # for a short clip must NOT trigger a re-sample.
    caps, segs = _caps_segs(3)
    client = _FakeClient([6])
    out = generate_emotion_events("v", caps, client, segs)
    assert client.calls == 1
    assert len(out.events) == 6


# --- merge_contiguous_events (Change 3: deterministic merge backstop) --------

def test_merge_adjacent_same_label_same_target():
    events = [
        _ev("e1", "angry", 10.0, 15.0, confidence="medium", evidence_strength="ambiguous",
            visual_evidence=["clenched jaw"], audio_evidence=[]),
        _ev("e2", "angry", 15.0, 20.0, confidence="high", evidence_strength="clear",
            visual_evidence=["raised voice"], audio_evidence=["shouting"]),
    ]
    merged = merge_contiguous_events(events, max_gap=5.0)
    assert len(merged) == 1
    m = merged[0]
    assert m.time_range == [10.0, 20.0]
    assert m.visual_evidence == ["clenched jaw", "raised voice"]
    assert m.audio_evidence == ["shouting"]
    # Keeps the higher confidence/evidence_strength across the merged pair.
    assert m.confidence == "high" and m.evidence_strength == "clear"


def test_does_not_merge_different_emotion_label():
    events = [
        _ev("e1", "angry", 10.0, 15.0),
        _ev("e2", "sad", 15.0, 20.0),
    ]
    merged = merge_contiguous_events(events, max_gap=5.0)
    assert len(merged) == 2


def test_does_not_merge_different_target():
    events = [
        _ev("e1", "angry", 10.0, 15.0, target="the man in the blue jacket"),
        _ev("e2", "angry", 15.0, 20.0, target="the woman in the red dress"),
    ]
    merged = merge_contiguous_events(events, max_gap=5.0)
    assert len(merged) == 2


def test_does_not_merge_non_contiguous_events():
    events = [
        _ev("e1", "angry", 10.0, 15.0),
        _ev("e2", "angry", 40.0, 45.0),  # far gap, well beyond max_gap
    ]
    merged = merge_contiguous_events(events, max_gap=5.0)
    assert len(merged) == 2


def test_does_not_merge_across_one_missing_segment():
    # A whole 5s segment between the two events produced no event (neutral):
    # under the production gap (half a segment) they must NOT merge, else the
    # emotion's time_range is falsely stretched across the neutral stretch.
    events = [
        _ev("e1", "angry", 0.0, 5.0),
        _ev("e2", "angry", 10.0, 15.0),  # gap == 5.0 (one segment)
    ]
    merged = merge_contiguous_events(events, max_gap=2.5)
    assert len(merged) == 2


def test_merge_chains_across_more_than_two_events():
    events = [
        _ev("e1", "happy", 0.0, 5.0),
        _ev("e2", "happy", 5.0, 10.0),
        _ev("e3", "happy", 10.0, 15.0),
    ]
    merged = merge_contiguous_events(events, max_gap=5.0)
    assert len(merged) == 1
    assert merged[0].time_range == [0.0, 15.0]
