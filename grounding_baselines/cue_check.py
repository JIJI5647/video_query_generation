"""Cross-model fine-grained cue verification (Qwen3-VL, transformers).

Tests whether an independent vision-strong model, prompted to be skeptical, can
flag the caption-stage hallucinations behind the 6% verifier false-pass — the
cases where Qwen3-Omni's caption invented a fine-grained visual cue (a single
tear, widened eyes) that isn't in the video. Qwen3-VL did NOT write these
captions, so it is a genuine second opinion.

For each cited visual cue + its segment clip(s), ask Qwen3-VL a yes/no:
"Is <cue> ACTUALLY visible in this clip?" — verdict NO = flagged as hallucinated.
Goal: flag the 6 HALLUC without flagging the GOOD controls.

  python grounding_baselines/cue_check.py --set output/grounding_eval/cue_check_set.jsonl \
      --model Qwen/Qwen3-VL-8B-Instruct --output output/grounding_eval/cue_check
"""
import argparse, json, os, re, sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROMPT = (
    "You are a strict visual fact-checker. Watch this short video clip carefully. "
    "Claim: \"{cue}\". Is this claim ACTUALLY, clearly visible in the clip? Do not "
    "guess or infer from context — only answer YES if you can directly see it. "
    "Answer strictly 'YES' or 'NO', then a short reason.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="output/grounding_eval/cue_check_set.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--output", required=True)
    ap.add_argument("--fps", type=float, default=2.0)
    args = ap.parse_args()

    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText as Cls
    from qwen_vl_utils import process_vision_info

    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(args.set) if l.strip()]
    model = Cls.from_pretrained(args.model, dtype=torch.bfloat16,
                                attn_implementation="sdpa", device_map="cuda")
    processor = AutoProcessor.from_pretrained(args.model)

    def ask(clip, cue):
        messages = [{"role": "user", "content": [
            {"type": "video", "video": f"file://{Path(clip).resolve()}", "fps": args.fps},
            {"type": "text", "text": PROMPT.format(cue=cue)}]}]
        text = processor.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)
        imgs, vids, vkw = process_vision_info(messages, return_video_kwargs=True,
                                              return_video_metadata=True)
        if vids:
            pairs = list(zip(*vids)); vids, meta = list(pairs[0]), list(pairs[1])
        else:
            meta = None
        inp = processor(text=[text], images=imgs, videos=vids, video_metadata=meta,
                        return_tensors="pt", **vkw).to("cuda")
        with torch.inference_mode():
            gen = model.generate(**inp, max_new_tokens=96, do_sample=False)
        return processor.batch_decode(gen[:, inp["input_ids"].shape[1]:],
                                      skip_special_tokens=True)[0].strip()

    results = []
    with open(out / "cue_check_results.jsonl", "w") as f:
        for r in rows:
            cue_text = "; ".join(r["cue"]) if isinstance(r["cue"], list) else str(r["cue"])
            # check each clip; cue counts as "visible" if ANY clip says YES
            per_clip = []
            for clip in r["clips"]:
                reply = ask(clip, cue_text)
                verdict = "YES" if re.match(r"\s*yes", reply, re.I) else "NO"
                per_clip.append({"clip": Path(clip).name, "verdict": verdict,
                                 "reply": reply[:120]})
            visible = any(c["verdict"] == "YES" for c in per_clip)
            flagged = not visible  # NO on all clips -> flagged as hallucinated
            row = {"query_id": r["query_id"], "label": r["label"], "cue": cue_text,
                   "flagged_hallucination": flagged, "per_clip": per_clip}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            mark = "✓" if (flagged == (r["label"] == "HALLUC")) else "✗"
            print(f"{mark} [{r['label']:<6}] {r['query_id'][:36]:<38} "
                  f"flagged={flagged}  cue={cue_text[:42]!r}", flush=True)
            results.append(row)

    # confusion
    tp = sum(1 for r in results if r["label"] == "HALLUC" and r["flagged_hallucination"])
    fn = sum(1 for r in results if r["label"] == "HALLUC" and not r["flagged_hallucination"])
    fp = sum(1 for r in results if r["label"] == "GOOD" and r["flagged_hallucination"])
    tn = sum(1 for r in results if r["label"] == "GOOD" and not r["flagged_hallucination"])
    print(f"\nHALLUC caught {tp}/{tp+fn} | GOOD wrongly flagged {fp}/{fp+tn}")


if __name__ == "__main__":
    main()
