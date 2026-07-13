"""Apply ONLY the re-grounding stage to a FIXED, pre-generated query set.

Controlled-experiment helper: load an existing initial_queries.jsonl (the clean
generation output, no regrounding), re-ground it with the chosen scope, and write
a new queries dir (initial_queries.jsonl + segments.jsonl) that run_verification.py
can consume. This lets off/full/window be compared on the SAME queries — the
generation stage is NOT re-run, so query text + generation grounding are identical
across scopes.

  python apply_regrounding.py --base-queries output/prompt_sweep/v12_off \
      --captions-dir output/exp3_unified/captions/qwen3_omni \
      --scope full --output output/sweep_fixed/v12_full
"""
import argparse, json, os, sys, shutil
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from emotion_query_pipeline.models import OmniCaption, EventGroundedQuery
from emotion_query_pipeline.regrounding import reground_queries
from emotion_query_pipeline.llm_client import GeminiLLMClient
from rerun_generation import _load_by_video, _load_segments_by_video


def _load_queries_by_video(path: Path):
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            vid = rec["video_id"]
            out[vid] = [EventGroundedQuery.model_validate(q) for q in rec.get("queries", [])]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-queries", required=True,
                    help="Dir with the FIXED initial_queries.jsonl to re-ground.")
    ap.add_argument("--captions-dir", required=True,
                    help="Dir with raw_captions.jsonl + segments.jsonl.")
    ap.add_argument("--scope", choices=["off", "full", "window"], required=True)
    ap.add_argument("--window", type=int, default=2)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    base = Path(args.base_queries)
    caps_dir = Path(args.captions_dir)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)

    queries_by_v = _load_queries_by_video(base / "initial_queries.jsonl")
    raw = _load_by_video(caps_dir / "raw_captions.jsonl", OmniCaption)
    segs = _load_segments_by_video(caps_dir / "segments.jsonl", set(raw.keys()))

    if args.scope == "off":
        # No re-grounding: the base queries ARE the result (generation grounding).
        print(f"[apply_reground] scope=off -> passthrough {sum(len(v) for v in queries_by_v.values())} queries")
    else:
        api_key = os.environ["GEMINI_API_KEY"]
        client = GeminiLLMClient(caption_model="gemini-2.5-flash-lite",
                                 generation_model="gemini-2.5-flash-lite",
                                 verification_model="gemini-2.5-flash-lite",
                                 rewrite_model="gemini-2.5-flash-lite",
                                 emotion_event_model=None, api_key=api_key)
        tot_ch = tot_fb = tot = 0
        for vid in sorted(queries_by_v):
            qs = queries_by_v[vid]
            if not qs:
                continue
            qs, st = reground_queries(vid, qs, raw.get(vid, []), segs.get(vid, []),
                                      client, scope=args.scope, window=args.window)
            queries_by_v[vid] = qs
            tot_ch += st["changed"]; tot_fb += st["fallback"]; tot += st["total"]
            print(f"  {vid}: {st['changed']} changed, {st['fallback']} fallback / {st['total']}", flush=True)
        print(f"[apply_reground] scope={args.scope}: {tot_ch}/{tot} changed, {tot_fb} fell back", flush=True)

    # Write the re-grounded queries back in initial_queries.jsonl format + copy segments.
    with open(out / "initial_queries.jsonl", "w") as f:
        for vid in sorted(queries_by_v):
            rec = {"video_id": vid,
                   "queries": [q.model_dump() for q in queries_by_v[vid]]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    # Copy segments from the BASE queries dir (its records carry video_id, which
    # run_verification requires; the captions-dir segments.jsonl does not).
    shutil.copy(base / "segments.jsonl", out / "segments.jsonl")
    print(f"[apply_reground] wrote {out}/initial_queries.jsonl + segments.jsonl", flush=True)


if __name__ == "__main__":
    main()
