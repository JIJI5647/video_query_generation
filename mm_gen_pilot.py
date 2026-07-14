"""Pilot: MULTIMODAL query generation with local Qwen3-Omni-30B (vLLM endpoint).

Controlled comparison of the GENERATION step only: takes the SAME emotion events
that Arm A (Gemini text-only generation) used, but lets Qwen3-Omni WATCH each
event's clip(s) and write the queries. time_range/segment_ids are inherited from
the event (fuzzy-time responsibility unchanged). Output is written in the
initial_queries.jsonl format so run_verification.py can verify it with the same
p7_rolecot backend, giving a pass-rate comparison against Arm A on the same
videos/events.

  python mm_gen_pilot.py --videos v1,v2,v3 --output output/mm_gen_pilot/qwen3_omni
"""
import argparse, json, os, sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from emotion_query_pipeline.nemotron_client import NemotronOpenAIClient

SRC = Path("output/eval_unified19/qwen3_omni")  # default; override with --events-dir

PROMPT = """You are writing emotion-related temporal grounding queries for human annotation.

Watch the provided video clip(s) carefully (they have audio - listen too). This moment
was flagged as an emotion event:
- target emotion: {emotion_label}
- involved person/group: {target}
- cited cues: visual={visual_evidence} audio={audio_evidence}

Write 1-3 queries of DIFFERENT query_types about this exact moment:
- emotion_state: the person's perceived emotional state (name the emotion, e.g. "appear {emotion_label}")
- evidence_cue: an observable cue tied to the emotion (e.g. "widen her eyes and gasp in surprise")
- explicit_event: an emotional action/interaction tied to the emotion

STRICT RULES:
0. QUERY FORM: every query_text MUST be a TEMPORAL QUESTION that asks WHEN something
   happens - start with "When does...", "At what point...", or "Which moment shows...".
   NEVER a yes/no question ("Does she..."), NEVER a statement, NEVER an instruction
   ("Identify..."). Example: "When does the woman in a white shirt widen her eyes
   and gasp in surprise?"
1. ANCHOR every query to the target emotion "{emotion_label}" - name it or use a cue that
   canonically signals it. A query that cannot be mapped to exactly this emotion is INVALID.
2. ONLY describe what you can ACTUALLY SEE or HEAR in the provided clip(s). If a cited cue
   is NOT visible/audible in the clip, do NOT use it. If the clip does not clearly support
   the emotion at all, return an empty queries list instead of inventing.
3. Each query_text must be SELF-CONTAINED: describe the person naturally (clothing, role,
   position), never segment ids, person ids, timestamps, or proper names.
4. Never quote spoken words (no transcript).
5. Fewer, well-supported queries beat padded weak ones.

Return ONLY valid JSON, no markdown fences:
{{
  "queries": [
    {{
      "query_type": "<emotion_state | evidence_cue | explicit_event>",
      "query_text": "<natural English query>",
      "grounding_evidence": {{
        "visual_evidence": ["<cue you actually saw>"],
        "audio_evidence": ["<cue you actually heard>"]
      }}
    }}
  ]
}}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True, help="comma-separated video_ids")
    ap.add_argument("--output", required=True)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--events-dir", default=None, help="dir with emotion_events.jsonl+segments.jsonl (default: eval_unified19/qwen3_omni)")
    ap.add_argument("--max-clips", type=int, default=2,
                    help="max clips fed per event (long merged events)")
    args = ap.parse_args()

    global SRC
    if args.events_dir:
        SRC = Path(args.events_dir)
    vids = [v.strip() for v in args.videos.split(",") if v.strip()]
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)

    # segments: (video_id, segment_id) -> clip_path ; also keep rows for copy
    seg_rows = [json.loads(l) for l in open(SRC / "segments.jsonl") if l.strip()]
    clip = {(r["video_id"], r["segment_id"]): r["clip_path"] for r in seg_rows}

    events = [json.loads(l) for l in open(SRC / "emotion_events.jsonl") if l.strip()
              if json.loads(l)["video_id"] in vids]
    print(f"[mm-gen] {len(events)} events over {len(vids)} videos")

    client = NemotronOpenAIClient(base_url=args.base_url, model=args.model,
                                  max_tokens=args.max_tokens, enable_thinking=None,
                                  use_audio_in_video=True, max_workers=4)

    by_vid = defaultdict(list)
    counter = defaultdict(int)
    n_ok = n_empty = n_err = 0
    for i, e in enumerate(events, 1):
        vid = e["video_id"]
        sids = e.get("segment_ids") or []
        clips = [clip[(vid, s)] for s in sids if (vid, s) in clip][: args.max_clips]
        if not clips:
            print(f"  [{i}/{len(events)}] {e['event_id']} no clips - skip"); continue
        prompt = PROMPT.format(
            emotion_label=e["emotion_label"],
            target=e.get("target_person_or_group", ""),
            visual_evidence=json.dumps(e.get("visual_evidence") or []),
            audio_evidence=json.dumps(e.get("audio_evidence") or []),
        )
        try:
            res = client.generate_json(prompt, "MMGenQueries", video_uri=clips)
        except Exception as ex:
            n_err += 1
            print(f"  [{i}/{len(events)}] {e['event_id']} ERROR: {str(ex)[:80]}")
            continue
        qs = res.get("queries") or []
        if not qs:
            n_empty += 1
            print(f"  [{i}/{len(events)}] {e['event_id']} -> 0 queries (declined)")
            continue
        for q in qs:
            counter[vid] += 1
            by_vid[vid].append({
                "video_id": vid,
                "query_id": f"{vid}_mm{counter[vid]:03d}",
                "query_type": q.get("query_type", ""),
                "query_text": q.get("query_text", ""),
                "time_range": e.get("time_range"),
                "segment_ids": sids,
                "grounding_evidence": q.get("grounding_evidence") or {},
                "source_caption_ids": [],
                "grounding_event_description": e.get("event_description", ""),
                "approximate_grounding_time": None,
                "target_person_or_group": e.get("target_person_or_group", ""),
                "expected_evidence": [],
                "why_grounded": "",
            })
        n_ok += 1
        print(f"  [{i}/{len(events)}] {e['event_id']} ({e['emotion_label']}) -> {len(qs)} queries")

    with open(out / "initial_queries.jsonl", "w") as f:
        for vid in vids:
            if by_vid[vid]:
                f.write(json.dumps({"video_id": vid, "queries": by_vid[vid]},
                                   ensure_ascii=False) + "\n")
    with open(out / "segments.jsonl", "w") as f:
        for r in seg_rows:
            if r["video_id"] in vids:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    total = sum(counter.values())
    print(f"\n[mm-gen] DONE: {total} queries from {n_ok} events "
          f"({n_empty} declined, {n_err} errors) -> {out}/initial_queries.jsonl")


if __name__ == "__main__":
    main()
