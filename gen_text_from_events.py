"""Gemini text v12 generation from a given emotion_events dir (baseline arm).

  python gen_text_from_events.py --events-dir <dir> --videos v1,v2 --output <dir>
"""
import argparse, json, os, shutil, sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from emotion_query_pipeline.models import OmniCaption, EmotionEvent
from emotion_query_pipeline.generation import generate_queries
from emotion_query_pipeline.llm_client import GeminiLLMClient
from rerun_generation import _load_by_video, _load_segments_by_video

CAP = Path("output/eval_unified19/qwen3_omni")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events-dir", required=True)
    ap.add_argument("--videos", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--gen-model", default="gemini-2.5-flash-lite")
    ap.add_argument("--captions-dir", default=str(CAP),
                    help="dir with raw_captions.jsonl + segments.jsonl "
                         "(default: eval_unified19/qwen3_omni)")
    args = ap.parse_args()

    vids = [v.strip() for v in args.videos.split(",") if v.strip()]
    evd = Path(args.events_dir)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)

    cap_dir = Path(args.captions_dir)
    raw = _load_by_video(cap_dir / "raw_captions.jsonl", OmniCaption)
    segs = _load_segments_by_video(cap_dir / "segments.jsonl", set(raw.keys()))
    ev_by = defaultdict(list)
    for l in open(evd / "emotion_events.jsonl"):
        d = json.loads(l)
        d.pop("target_person", None); d.pop("segment_ids", None)
        ev_by[d["video_id"]].append(EmotionEvent.model_validate(d))

    client = GeminiLLMClient(
        caption_model=args.gen_model, generation_model=args.gen_model,
        verification_model=args.gen_model, rewrite_model=args.gen_model,
        emotion_event_model=None, api_key=os.environ["GEMINI_API_KEY"])

    with open(out / "initial_queries.jsonl", "w") as f:
        for vid in vids:
            if not ev_by.get(vid):
                print(f"{vid}: 0 events - skip", flush=True); continue
            try:
                gen = generate_queries(vid, raw[vid], ev_by[vid], client,
                                       segs.get(vid, []))
            except Exception as ex:
                print(f"{vid}: ERROR {str(ex)[:80]}", flush=True); continue
            f.write(json.dumps({"video_id": vid, "queries":
                    [q.model_dump() for q in gen.queries]},
                    ensure_ascii=False) + "\n")
            print(f"{vid}: {len(gen.queries)} queries", flush=True)
    shutil.copy(evd / "segments.jsonl", out / "segments.jsonl")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
