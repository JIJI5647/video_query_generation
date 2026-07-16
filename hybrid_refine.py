"""Hybrid gen: Gemini writes disambiguated queries, a watch model (Nemotron or
Qwen3-Omni) grounds+verifies each against its actual clip.

The watch model's job is NOT to invent detail (that is where the 6% caption
hallucination came from) but to CHECK the claimed cue against the pixels: confirm
+ make concrete what it sees, mark unsupported what it doesn't.

  python hybrid_refine.py --base output/hybrid_proto/gemini_base \
      --model nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8 \
      --output output/hybrid_proto/refine_nemotron
"""
import argparse, json, glob
from pathlib import Path
from openai import OpenAI

PROMPT = """You are grounding an emotion-related temporal query against the ACTUAL video clip(s). Watch carefully (they have audio - listen too).

QUERY: "{query_text}"
It relies on these claimed cues: visual={visual_evidence} audio={audio_evidence}

Do TWO things:
1. VERIFY: is the described moment/cue ACTUALLY visible or audible in this clip? Only say supported=true if you can directly see/hear it. If the key cue is NOT there, supported=false.
2. GROUND (only if supported): rewrite the query to be MORE concrete using ONE specific detail you ACTUALLY see or hear (a precise expression, gesture, action, or sound). Keep it a temporal question ("When does...", "At what point...", "Which moment shows..."), keep the SAME person and SAME target emotion, add no proper names/ids/timestamps. If there is nothing concrete to add, return the query unchanged.

Do NOT invent details that are not in the clip. Return ONLY JSON:
{{"supported": true/false, "refined_query_text": "<grounded query or original>", "observed": "<the concrete detail you saw/heard, or empty>"}}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="dir with initial_queries.jsonl + segments.jsonl")
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--thinking", choices=["true", "false", "none"], default="false")
    ap.add_argument("--max-clips", type=int, default=2)
    args = ap.parse_args()

    base = Path(args.base)
    seg_rows = [json.loads(l) for l in open(base / "segments.jsonl") if l.strip()]
    clip = {(r["video_id"], r["segment_id"]): r["clip_path"] for r in seg_rows}

    def clips_for(vid, sids):
        return [f"file://{Path(clip[(vid,s)]).resolve()}"
                for s in sids if (vid, s) in clip][:args.max_clips]

    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=600)
    extra = {"mm_processor_kwargs": {"use_audio_in_video": True}}
    if args.thinking != "none":
        extra["chat_template_kwargs"] = {"enable_thinking": args.thinking == "true"}

    rows = [json.loads(l) for l in open(base / "initial_queries.jsonl") if l.strip()]
    n_in = n_sup = n_unsup = n_changed = n_err = 0
    fout = open(out / "refined_queries.jsonl", "w")
    for row in rows:
        vid = row["video_id"]
        kept = []
        for q in row.get("queries", []):
            n_in += 1
            sids = q.get("segment_ids") or []
            uris = clips_for(vid, sids)
            ge = q.get("grounding_evidence") or {}
            if not uris:
                kept.append({**q, "_verdict": "no_clip"}); continue
            prompt = PROMPT.format(query_text=q["query_text"],
                visual_evidence=json.dumps(ge.get("visual_evidence") or []),
                audio_evidence=json.dumps(ge.get("audio_evidence") or []))
            content = [{"type": "video_url", "video_url": {"url": u}} for u in uris]
            content.append({"type": "text", "text": prompt})
            try:
                r = client.chat.completions.create(model=args.model,
                    messages=[{"role": "user", "content": content}],
                    max_tokens=args.max_tokens, temperature=0.0, extra_body=extra)
                txt = (r.choices[0].message.content or "").strip()
                if "</think>" in txt: txt = txt.split("</think>")[-1].strip()
                s = txt.find("{"); d = json.loads(txt[s:txt.rfind("}")+1])
            except Exception as ex:
                n_err += 1
                kept.append({**q, "_verdict": f"err:{str(ex)[:40]}"}); continue
            if d.get("supported"):
                n_sup += 1
                new = (d.get("refined_query_text") or "").strip() or q["query_text"]
                if new != q["query_text"]: n_changed += 1
                kept.append({**q, "query_text": new, "_verdict": "supported",
                             "_orig": q["query_text"], "_observed": d.get("observed", "")})
            else:
                n_unsup += 1
                # dropped from kept; log it
                kept.append({**q, "_verdict": "unsupported", "_dropped": True})
        fout.write(json.dumps({"video_id": vid, "queries": kept}, ensure_ascii=False) + "\n")
        fout.flush()
        print(f"{vid}: {len(kept)} processed", flush=True)
    fout.close()
    print(f"\nIN {n_in} | supported {n_sup} (changed {n_changed}) | "
          f"unsupported/dropped {n_unsup} | err {n_err}", flush=True)


if __name__ == "__main__":
    main()
