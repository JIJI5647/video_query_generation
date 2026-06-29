"""Convert a human verification-annotation export into usable eval inputs.

Input: the ``*_final_query_verification_annotations.json`` export (keys: schema,
annotations[...]), where each annotation has the query, its segment_ids /
approximate_time, and human_* + machine_* pass/fail labels.

Outputs (into --output dir):
  - gold.jsonl            one row per query with the HUMAN labels + derived gold
                          decision (queries with missing human labels are kept
                          with gold_decision=null so you can filter them out).
  - initial_queries.jsonl GenerationOutput per video — feed to run_verification.py
                          (--queries-dir) to re-verify these exact queries.
  - segments.jsonl        Segment per (video, segment) parsed from the ids/times.

Usage:
    python annotations_to_eval.py \
        --annotations data/test5_..._annotations.json \
        --output data/test5_eval
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_TIME_RE = re.compile(r"\(([\d.]+)\s*-\s*([\d.]+)\s*s\)")
_SEG_NUM_RE = re.compile(r"s(\d+)")


def _decision(rel: bool, ans: bool, qual: bool) -> str:
    """Same routing the pipeline uses to turn 3 dimensions into a decision."""
    if not rel or not ans:
        return "fail"
    if not qual:
        return "revise"
    return "pass"


def _seg_start_end(segment_id: str, seconds: float = 5.0):
    """Derive [start, end] for a segment id like 's002' on a fixed-length grid."""
    m = _SEG_NUM_RE.search(segment_id)
    n = int(m.group(1)) if m else 1
    return (n - 1) * seconds, n * seconds, n


def main() -> None:
    ap = argparse.ArgumentParser(description="Annotations -> eval inputs.")
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--segment-seconds", type=float, default=5.0)
    args = ap.parse_args()

    data = json.loads(Path(args.annotations).read_text(encoding="utf-8"))
    annotations = data.get("annotations") or []
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Authoritative per-segment [start, end] from SINGLE-segment queries: their
    # approximate_time carries the real range, including a short final segment at
    # the video's tail (e.g. "s011 (50-54.4s)"). The fixed grid is only a fallback
    # and would overshoot the last segment past the end of the video.
    seg_exact: dict = {}
    for a in annotations:
        sids = a.get("segment_ids") or []
        m = _TIME_RE.search(a.get("approximate_time") or "")
        if len(sids) == 1 and m:
            seg_exact[(a["video_id"], sids[0])] = (float(m.group(1)), float(m.group(2)))

    gold_rows = []
    queries_by_video: dict = {}
    segments_by_video: dict = {}
    labeled = 0

    for a in annotations:
        vid = a["video_id"]
        qid = a["query_id"]
        text = a.get("final_query_text") or a.get("initial_query_text") or ""
        seg_ids = list(a.get("segment_ids") or [])

        # Query time range: parse "(5-10s)" from approximate_time, else from ids.
        tr = None
        m = _TIME_RE.search(a.get("approximate_time") or "")
        if m:
            tr = [float(m.group(1)), float(m.group(2))]
        elif seg_ids:
            starts, ends = [], []
            for sid in seg_ids:
                s, e, _ = _seg_start_end(sid, args.segment_seconds)
                starts.append(s); ends.append(e)
            tr = [min(starts), max(ends)]

        # --- initial_queries.jsonl (per video) ---
        queries_by_video.setdefault(vid, []).append({
            "video_id": vid,
            "query_id": qid,
            "query_type": a.get("query_type") or "emotion_state",
            "query_text": text,
            "time_range": tr,
            "segment_ids": seg_ids,
        })

        # --- segments.jsonl (per video, dedup) ---
        seg_map = segments_by_video.setdefault(vid, {})
        for sid in seg_ids:
            if sid not in seg_map:
                grid_s, grid_e, n = _seg_start_end(sid, args.segment_seconds)
                # Prefer the exact range from a single-segment query (handles the
                # short final segment); fall back to the grid otherwise.
                s, e = seg_exact.get((vid, sid), (grid_s, grid_e))
                seg_map[sid] = {
                    "video_id": vid, "segment_id": sid, "index": n,
                    "start_time": s, "end_time": e, "clip_path": None,
                }

        # --- gold.jsonl ---
        hr, ha, hq = (a.get("human_emotion_relevant"),
                      a.get("human_answerability"),
                      a.get("human_query_quality"))
        if hr in ("pass", "fail") and ha in ("pass", "fail") and hq in ("pass", "fail"):
            rel, ans, qual = hr == "pass", ha == "pass", hq == "pass"
            gold_dec = _decision(rel, ans, qual)
            labeled += 1
        else:
            rel = ans = qual = None
            gold_dec = None
        gold_rows.append({
            "video_id": vid,
            "query_id": qid,
            "query_text": text,
            "segment_ids": seg_ids,
            "gold_emotion_relevant": rel,
            "gold_answerability": ans,
            "gold_query_quality": qual,
            "gold_decision": gold_dec,
        })

    def _write(name, rows):
        path = out_dir / name
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return path

    _write("gold.jsonl", gold_rows)
    _write("initial_queries.jsonl",
           [{"video_id": v, "queries": q} for v, q in queries_by_video.items()])
    _write("segments.jsonl",
           [s for segs in segments_by_video.values() for s in segs.values()])

    print(f"Wrote to {out_dir}/")
    print(f"  gold.jsonl            : {len(gold_rows)} rows "
          f"({labeled} fully human-labeled, {len(gold_rows) - labeled} unlabeled)")
    print(f"  initial_queries.jsonl : {len(queries_by_video)} video(s), "
          f"{sum(len(q) for q in queries_by_video.values())} queries")
    print(f"  segments.jsonl        : "
          f"{sum(len(s) for s in segments_by_video.values())} segments")


if __name__ == "__main__":
    main()
