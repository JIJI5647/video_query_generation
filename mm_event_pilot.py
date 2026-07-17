"""Event-stage pilot: v2 rules, text vs MULTIMODAL grounding, several models.

Produces emotion_events.jsonl (+ segments.jsonl) per arm for the same pilot
videos, with v2 CODE-SIDE validation/normalization applied uniformly:
  - evidence items parsed for their [sNNN] source prefix
  - event time_range forced to cover cited segments (contiguous span);
    non-conforming cues dropped
  - span capped at 4 segments (cues outside the best 4-seg window dropped)
  - confidence=low / evidence_strength=weak events dropped
  - target_person mapped to target_person_or_group for downstream compat

Backends:
  gemini-text : Gemini reads captions (prompts/emotion_event_prompt_v2.txt)
  gemini-mm   : Gemini WATCHES the full video (prompts/emotion_event_prompt_v2_mm.txt)
  local       : vLLM OpenAI endpoint model WATCHES the full video (same mm prompt)

  python mm_event_pilot.py --backend gemini-text --output output/mm_event_pilot/T2 ...
"""
import argparse, json, os, re, sys, time
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SRC = Path("output/eval_unified19/qwen3_omni")   # captions + segments source
VIDEO_DIR = Path("data/pilot_study")

SEG_RE = re.compile(r"^\s*\[(s\d+)\]\s*")


def load_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def caption_payload(caps):
    out = []
    for c in caps:
        v = (c.get("visual_description") or "").strip()
        a = (c.get("audio_description") or "").strip()
        txt = f"Visual: {v}" + (f"\nAudio: {a}" if a else "")
        out.append({"segment_id": c["segment_id"], "time_range": c["time_range"],
                    "caption": txt, "confidence": c.get("confidence"),
                    "evidence_strength": c.get("evidence_strength")})
    return out


def normalize_v2(events, seg_time, video_id):
    """Apply v2 code-side guards; returns (kept_events, stats)."""
    stats = {"in": len(events), "dropped_lowconf": 0, "cues_dropped": 0,
             "range_fixed": 0, "kept": 0}
    idx_of = {s: int(s[1:]) for s in seg_time}
    by_idx = {int(s[1:]): s for s in seg_time}
    kept = []
    for k, e in enumerate(events):
        conf = (e.get("confidence") or "").lower()
        strength = (e.get("evidence_strength") or "").lower()
        if conf == "low" or strength == "weak":
            stats["dropped_lowconf"] += 1
            continue
        # parse cited segments from evidence prefixes
        cited = []
        for key in ("visual_evidence", "audio_evidence"):
            for item in e.get(key) or []:
                m = SEG_RE.match(str(item))
                if m and m.group(1) in seg_time:
                    cited.append(m.group(1))
        if cited:
            idxs = sorted({idx_of[s] for s in cited})
            # NO CAP: cover the full contiguous span of the cited segments so a
            # long continuous emotion (e.g. a 50s cry) is not truncated. The v2
            # prompt still splits on label/person/gap; this only removes the
            # former <=4-segment ceiling.
            lo = idxs[0]; hi = idxs[-1]
            win = {i for i in range(lo, hi + 1) if i in by_idx}
            keep_ids = {by_idx[i] for i in win}
            # drop cues outside the window
            for key in ("visual_evidence", "audio_evidence"):
                orig = e.get(key) or []
                kept_items = []
                for item in orig:
                    m = SEG_RE.match(str(item))
                    if m and m.group(1) in seg_time and m.group(1) not in keep_ids:
                        stats["cues_dropped"] += 1
                        continue
                    kept_items.append(item)
                e[key] = kept_items
            covered = sorted(win)
            new_ids = [by_idx[i] for i in covered]
            t0 = min(seg_time[s][0] for s in new_ids)
            t1 = max(seg_time[s][1] for s in new_ids)
            if e.get("time_range") != [t0, t1]:
                stats["range_fixed"] += 1
            e["time_range"] = [t0, t1]
            e["segment_ids"] = new_ids
        else:
            # no parsable citations: keep only if time_range maps onto grid
            tr = e.get("time_range") or []
            if len(tr) != 2:
                continue
            ids = [s for s, (a, b) in seg_time.items() if a < tr[1] and tr[0] < b]
            ids = sorted(ids, key=lambda s: idx_of[s])
            if not ids:
                continue
            e["segment_ids"] = ids
            e["time_range"] = [seg_time[ids[0]][0], seg_time[ids[-1]][1]]
        e["video_id"] = video_id
        e["event_id"] = e.get("event_id") or f"e{k}"
        e["target_person_or_group"] = e.get("target_person") or e.get(
            "target_person_or_group", "")
        kept.append(e)
    stats["kept"] = len(kept)
    return kept, stats


def main():
    global SRC
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True,
                    choices=["gemini-text", "gemini-mm", "local"])
    ap.add_argument("--videos", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="Qwen/Qwen3-Omni-30B-A3B-Instruct",
                    help="local backend served model / gemini model name")
    ap.add_argument("--gemini-model", default="gemini-2.5-flash")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--no-thinking", action="store_true")
    ap.add_argument("--captions-dir", default=str(SRC),
                    help="dir with raw_captions.jsonl + segments.jsonl "
                         "(default: eval_unified19/qwen3_omni)")
    args = ap.parse_args()

    SRC = Path(args.captions_dir)
    vids = [v.strip() for v in args.videos.split(",") if v.strip()]
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)

    seg_rows = [r for r in load_jsonl(SRC / "segments.jsonl") if r["video_id"] in vids]
    caps_by_vid = defaultdict(list)
    for c in load_jsonl(SRC / "raw_captions.jsonl"):
        if c["video_id"] in vids:
            caps_by_vid[c["video_id"]].append(c)

    if args.backend == "gemini-text":
        tmpl = open("prompts/emotion_event_prompt_v2.txt").read()
    else:
        tmpl = open("prompts/emotion_event_prompt_v2_mm.txt").read()

    all_events = []
    for vid in vids:
        segs = {r["segment_id"]: (r["start_time"], r["end_time"])
                for r in seg_rows if r["video_id"] == vid}
        grid = json.dumps([{"segment_id": s, "time_range": list(segs[s])}
                           for s in sorted(segs, key=lambda x: int(x[1:]))])
        prompt = tmpl.replace("{video_id}", vid)
        if args.backend == "gemini-text":
            payload = json.dumps(caption_payload(caps_by_vid[vid]),
                                 ensure_ascii=False, indent=1)
            prompt = prompt.replace("{captions_json}", payload)
        else:
            prompt = prompt.replace("{segment_grid}", grid)

        t0 = time.time()
        if args.backend == "local":
            import subprocess, tempfile
            from emotion_query_pipeline.nemotron_client import NemotronOpenAIClient
            client = NemotronOpenAIClient(
                base_url=args.base_url, model=args.model,
                max_tokens=args.max_tokens,
                enable_thinking=(False if args.no_thinking else True),
                use_audio_in_video=True, max_workers=1)
            # windowed: cut <=60s chunks (context-safe), per-window grid, merge
            video_path = str(VIDEO_DIR / f"{vid}.mp4")
            ordered = sorted(segs, key=lambda x: int(x[1:]))
            win_events = []
            W = 4   # 4 x 5s = 20s per window (context-safe)
            for w0 in range(0, len(ordered), W):
                wseg = ordered[w0:w0 + W]
                t0w = segs[wseg[0]][0]; t1w = segs[wseg[-1]][1]
                wgrid = json.dumps([{"segment_id": s2,
                                     "time_range": list(segs[s2])} for s2 in wseg])
                wprompt = tmpl.replace("{video_id}", vid).replace(
                    "{segment_grid}", wgrid)
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
                    tmp = tf.name
                subprocess.run(["ffmpeg", "-y", "-loglevel", "error",
                                "-ss", str(t0w), "-to", str(t1w),
                                "-i", video_path, "-c", "copy", tmp], check=True)
                try:
                    wres = client.generate_json(wprompt, "MMEvents", video_uri=tmp)
                    win_events.extend(wres.get("events") or [])
                    print(f"  window {t0w:.0f}-{t1w:.0f}s: "
                          f"{len(wres.get('events') or [])} events", flush=True)
                except Exception as ex:
                    print(f"  window {t0w:.0f}-{t1w:.0f}s FAILED: "
                          f"{str(ex)[:60]}", flush=True)
                finally:
                    os.unlink(tmp)
            res = {"events": win_events}
        else:
            from google import genai
            g = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
            contents = [prompt]
            if args.backend == "gemini-mm":
                f = g.files.upload(file=str(VIDEO_DIR / f"{vid}.mp4"))
                while f.state.name == "PROCESSING":
                    time.sleep(3); f = g.files.get(name=f.name)
                contents = [f, prompt]
            res = None
            for attempt in range(3):
                r = g.models.generate_content(
                    model=args.gemini_model, contents=contents,
                    config={"response_mime_type": "application/json"})
                txt = (r.text or "").strip()
                txt = re.sub(r"^```(json)?|```$", "", txt, flags=re.M).strip()
                if "{" in txt and "}" in txt:
                    try:
                        res = json.loads(txt[txt.index("{"): txt.rindex("}") + 1])
                        break
                    except json.JSONDecodeError as ex:
                        print(f"  [retry {attempt+1}] JSON decode: {ex}", flush=True)
                else:
                    print(f"  [retry {attempt+1}] no JSON in response: "
                          f"{txt[:150]!r}", flush=True)
                time.sleep(3)
            if res is None:
                print(f"[{vid}] FAILED after retries - skipped", flush=True)
                continue
        raw = res.get("events") or []
        kept, st = normalize_v2(raw, segs, vid)
        all_events.extend(kept)
        print(f"[{vid}] raw={st['in']} kept={st['kept']} "
              f"(low丢{st['dropped_lowconf']}, cue丢{st['cues_dropped']}, "
              f"range修{st['range_fixed']}) {time.time()-t0:.0f}s", flush=True)

    with open(out / "emotion_events.jsonl", "w") as f:
        for e in all_events:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    with open(out / "segments.jsonl", "w") as f:
        for r in seg_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"DONE: {len(all_events)} events -> {out}/emotion_events.jsonl", flush=True)


if __name__ == "__main__":
    main()
