"""Per-dimension (decomposed) verification: composer + routing + merge."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from emotion_query_pipeline.models import EventGroundedQuery
from emotion_query_pipeline.verification import (
    _build_dim_prompt,
    verify_queries_per_dimension,
)

_VARIANTS = [
    "p0_norule", "p1_rule", "p2_role", "p3_fewshot", "p4_zscot",
    "p5_fewshotcot", "p6_rolefewshot", "p7_rolecot", "p8_rawcot",
]


def _q(qid="q1"):
    return EventGroundedQuery(
        video_id="v", query_id=qid, query_type="emotion_state", query_text="x"
    )


def test_composer_applies_variant_strategy():
    q = _q()
    p0 = _build_dim_prompt("relevance_pass", "v", q, 1, None, "p0_norule")
    p1 = _build_dim_prompt("relevance_pass", "v", q, 1, None, "p1_rule")
    p3 = _build_dim_prompt("relevance_pass", "v", q, 1, None, "p3_fewshot")
    p4 = _build_dim_prompt("relevance_pass", "v", q, 1, None, "p4_zscot")
    # p0 has no rule; p1+ include the rule. p3 adds examples; p4 adds CoT.
    assert "Mark relevance_pass" not in p0
    assert "Mark relevance_pass" in p1
    assert "EXAMPLES" in p3 and "EXAMPLES" not in p1
    assert "think step by step" in p4 and "brief reasoning before the JSON" in p4
    # every variant x dimension file exists, loads, and fully fills.
    for variant in _VARIANTS:
        for dim in ("relevance_pass", "answerability_pass", "query_quality_pass"):
            text = _build_dim_prompt(dim, "v", q, 1, None, variant)
            assert "{{include" not in text and "{queries_json}" not in text
            assert "{video_id}" not in text and dim in text


def test_text_vs_video_framing_per_dimension():
    q = _q()
    rel = _build_dim_prompt("relevance_pass", "v", q, 1, None, "p1_rule")
    ans = _build_dim_prompt("answerability_pass", "v", q, 1, None, "p1_rule")
    qual = _build_dim_prompt("query_quality_pass", "v", q, 1, None, "p1_rule")
    assert "QUERY TEXT ALONE" in rel and "Watch the provided video" not in rel
    assert "Watch the provided video" in ans
    assert "QUERY TEXT ALONE" in qual and "Watch the provided video" not in qual


class _FakeClient:
    """Returns each single-dim prompt's own field; records the video routing."""

    def __init__(self):
        self.routes = []

    def generate_json_many(self, prompts, schema_name, video_uris=None):
        out = []
        for p, u in zip(prompts, video_uris):
            if "Watch the provided video" in p:
                dim = "answerability_pass"
            elif "relevance_pass" in p.split("OUTPUT")[0] and "QUERY TEXT ALONE" in p:
                # relevance vs quality both text-only; disambiguate by output field
                dim = "relevance_pass" if '"relevance_pass"' in p.split("OUTPUT")[1] else "query_quality_pass"
            else:
                dim = "query_quality_pass"
            self.routes.append((dim, u))
            out.append({"results": [{"query_id": "q1", dim: True, "failure_reason": ""}]})
        return out


def test_routing_and_merge_pass():
    client = _FakeClient()
    out = verify_queries_per_dimension(
        "v", [_q()], [["clip.mp4"]], 1, client, variant="p1_rule"
    )
    routes = dict(client.routes)
    assert routes["relevance_pass"] is None  # text-only
    assert routes["query_quality_pass"] is None  # text-only
    assert routes["answerability_pass"] == ["clip.mp4"]  # watches clip
    assert out.results[0].decision == "pass"


def test_unknown_variant_raises():
    try:
        _build_dim_prompt("relevance_pass", "v", _q(), 1, None, "nope")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown variant")
