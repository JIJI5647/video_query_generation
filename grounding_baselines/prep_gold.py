"""Build the grounding-baseline gold set from the human annotation file.

Takes output/nemotron_watch_100.json (human 5s-segment annotations), drops
not_groundable, attaches video path + duration (ffprobe), writes
output/grounding_eval/gold.jsonl with one row per query:
  {query_id, video_id, video_path, duration, query_text, query_type,
   gold_ranges: [[s,e],...], model_time_range}
"""
import argparse, json, subprocess
from pathlib import Path


def merge_ranges(ranges, gap=0.0):
    """Merge overlapping/adjacent [s,e] ranges into maximal contiguous spans.
    The annotation tool stores each ticked 5s segment as its own range, so
    contiguous ticks arrive fragmented ([[15,20],[20,25]] = one 15-25 moment)."""
    rs = sorted([float(a), float(b)] for a, b in ranges)
    out = [rs[0]]
    for s, e in rs[1:]:
        if s <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def probe_duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True).stdout.strip()
    return float(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", default="output/nemotron_watch_100.json")
    ap.add_argument("--video-dir", default="data/pilot_study")
    ap.add_argument("--output", default="output/grounding_eval/gold.jsonl")
    args = ap.parse_args()

    ann = json.load(open(args.annotations))["annotations"]
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    durations = {}
    kept = dropped = 0
    with open(out, "w") as f:
        for a in ann:
            if a.get("not_groundable"):
                dropped += 1
                continue
            vid = a["video_id"]
            vpath = Path(args.video_dir) / f"{vid}.mp4"
            assert vpath.exists(), vpath
            if vid not in durations:
                durations[vid] = probe_duration(vpath)
            f.write(json.dumps({
                "query_id": a["query_id"],
                "video_id": vid,
                "video_path": str(vpath),
                "duration": round(durations[vid], 2),
                "query_text": a["query_text"],
                "query_type": a["query_type"],
                "gold_ranges": merge_ranges(a["human_time_ranges"]),
                "model_time_range": a.get("model_time_range"),
            }, ensure_ascii=False) + "\n")
            kept += 1
    print(f"kept {kept}, dropped(not_groundable) {dropped} -> {out}")


if __name__ == "__main__":
    main()
