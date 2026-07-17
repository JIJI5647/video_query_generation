"""A/B test: does text-based re-grounding (Gemini re-selects segments from
captions+query) improve timestamp agreement with human gold?

Runs reground_queries on the 50 human-annotated hybrid queries and compares the
re-grounded time_range vs the original proposal vs human gold. Text-only (no
video) — different from the MLLM-watch baselines.
"""
import json, os, sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from emotion_query_pipeline.models import OmniCaption, EventGroundedQuery
from emotion_query_pipeline.regrounding import reground_queries
from emotion_query_pipeline.llm_client import GeminiLLMClient
from rerun_generation import _load_by_video, _load_segments_by_video

CAP = Path("output/eval_unified19/qwen3_omni")


def merge(ranges):
    if not ranges:
        return []
    rs = sorted([float(a), float(b)] for a, b in ranges)
    out = [rs[0]]
    for s, e in rs[1:]:
        if s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def iou(a, b):
    if not a or not b:
        return 0.0
    i = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    u = max(a[1], b[1]) - min(a[0], b[0])
    return i / u if u > 0 else 0.0


def maxiou(p, rs):
    return max((iou(p, r) for r in rs), default=0.0) if (p and rs) else 0.0


def main():
    ann = json.load(open("output/hybrid_50.json"))["annotations"]
    ann_by_id = {a["query_id"]: a for a in ann}
    want = set(ann_by_id)

    # load the 50 queries as EventGroundedQuery, grouped by video
    raw = _load_by_video(CAP / "raw_captions.jsonl", OmniCaption)
    segs = _load_segments_by_video(CAP / "segments.jsonl", set(raw.keys()))
    q_by_vid = defaultdict(list)
    for row in (json.loads(l) for l in open("output/hybrid_full/final/initial_queries.jsonl")):
        for q in row["queries"]:
            if q["query_id"] in want:
                d = dict(q)
                d.pop("_verdict", None); d.pop("_orig", None); d.pop("_observed", None)
                q_by_vid[row["video_id"]].append(EventGroundedQuery.model_validate(d))

    client = GeminiLLMClient(
        caption_model="gemini-2.5-flash-lite", generation_model="gemini-2.5-flash-lite",
        verification_model="gemini-2.5-flash-lite", rewrite_model="gemini-2.5-flash-lite",
        emotion_event_model=None, api_key=os.environ["GEMINI_API_KEY"])

    regrounded = {}
    for vid, qs in q_by_vid.items():
        out, _stats = reground_queries(vid, qs, raw[vid], segs.get(vid, []), client, scope="full")
        for q in out:
            regrounded[q.query_id] = list(q.time_range) if q.time_range else None
        print(f"{vid}: {len(qs)} reground", flush=True)

    # compare on groundable subset
    orig_ious, rg_ious = [], []
    changed = better = worse = 0
    for qid, a in ann_by_id.items():
        if a.get("not_groundable"):
            continue
        hr = merge(a["human_time_ranges"])
        orig = [float(a["model_time_range"][0]), float(a["model_time_range"][1])]
        rg = regrounded.get(qid)
        io_o = maxiou(orig, hr); io_r = maxiou(rg, hr) if rg else io_o
        orig_ious.append(io_o); rg_ious.append(io_r)
        if rg and rg != orig:
            changed += 1
            if io_r > io_o + 0.02: better += 1
            elif io_o > io_r + 0.02: worse += 1
    n = len(orig_ious)
    print(f"\n==== text-based regrounding A/B (n={n}) ====")
    print(f"  原提案     mIoU {sum(orig_ious)/n:.3f} | R@0.5 {sum(1 for x in orig_ious if x>=.5)/n:.0%}")
    print(f"  regrounded mIoU {sum(rg_ious)/n:.3f} | R@0.5 {sum(1 for x in rg_ious if x>=.5)/n:.0%}")
    print(f"  改动 {changed} 条: 变好 {better}, 变差 {worse}")


if __name__ == "__main__":
    main()
