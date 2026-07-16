"""Sliding-window temporal grounding for Qwen3-VL (transformers).

Hypothesis (from docs/grounding_baseline_313_crossval.md): whole-video grounding
collapses on long videos because uniform frame subsampling destroys temporal
resolution (short<=60s mIoU 0.45 vs long>150s 0.16). Fix: scan the video in
short overlapping windows so each window is dense, ask each "is the moment here,
and where (within-window seconds)", map back to absolute time, aggregate.

  python grounding_baselines/run_window_scan.py \
      --model Qwen/Qwen3-VL-30B-A3B-Instruct \
      --gold output/grounding_eval/gold_long.jsonl \
      --output output/grounding_eval/qwen3vl30b_window \
      --window 60 --stride 40

Each window clip is cut with ffmpeg stream-copy into $CLAUDE_JOB_DIR/tmp.
Aggregation: merge overlapping present-spans; pick the cluster with the most
corroborating windows (ties -> longest span). Reuses parse_time_span.
"""
import argparse, json, os, re, subprocess, sys, tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from grounding_baselines.run_qwenvl import parse_time_span

WIN_PROMPT = (
    "This is a {wlen:.0f}-second clip. Give the query: '{query}'. "
    "Does the described content occur in THIS clip? If yes, answer with the "
    "start and end time in seconds measured from the start of THIS clip, in the "
    "format 'from X seconds to Y seconds'. If it does not occur in this clip, "
    "answer exactly 'not present'.")

TMP = Path(os.environ.get("CLAUDE_JOB_DIR", "/tmp")) / "tmp" / "winscan"


def cut(src, start, dur, dst):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", str(start), "-i", str(src),
                    "-t", str(dur), "-c", "copy", str(dst)], check=True)


def merge_cluster(spans):
    """spans: list of [s,e] absolute. Merge overlapping into clusters, return
    (best_span, support) where support = #spans in the largest cluster."""
    if not spans:
        return None, 0
    spans = sorted(spans)
    clusters = [[spans[0]]]
    for s in spans[1:]:
        if s[0] <= clusters[-1][-1][1]:  # overlaps current cluster
            clusters[-1].append(s)
        else:
            clusters.append([s])
    # largest cluster by count, tie -> longest merged span
    best = max(clusters, key=lambda c: (len(c), max(e for _, e in c) - min(s for s, _ in c)))
    lo = min(s for s, _ in best)
    hi = max(e for _, e in best)
    return [lo, hi], len(best)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--window", type=float, default=60.0)
    ap.add_argument("--stride", type=float, default=40.0)
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--max-pixels", type=int, default=360 * 420)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText as Cls
    from qwen_vl_utils import process_vision_info

    TMP.mkdir(parents=True, exist_ok=True)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    pred_path = out / "predictions.jsonl"
    done = {json.loads(l)["query_id"] for l in open(pred_path)} if pred_path.exists() else set()

    gold = [json.loads(l) for l in open(args.gold) if l.strip()]
    if args.limit:
        gold = gold[:args.limit]
    todo = [g for g in gold if g["query_id"] not in done]
    print(f"{len(todo)} queries ({len(done)} cached), W={args.window} S={args.stride}",
          flush=True)
    if not todo:
        return

    model = Cls.from_pretrained(args.model, dtype=torch.bfloat16,
                                attn_implementation="sdpa", device_map="cuda")
    processor = AutoProcessor.from_pretrained(args.model)

    def ask(clip_path, prompt):
        messages = [{"role": "user", "content": [
            {"type": "video", "video": f"file://{clip_path}", "fps": args.fps,
             "max_pixels": args.max_pixels},
            {"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)
        imgs, vids, vkw = process_vision_info(messages, return_video_kwargs=True,
                                              return_video_metadata=True)
        if vids:
            pairs = list(zip(*vids))
            vids, meta = list(pairs[0]), list(pairs[1])
        else:
            meta = None
        inp = processor(text=[text], images=imgs, videos=vids, video_metadata=meta,
                        return_tensors="pt", **vkw).to("cuda")
        with torch.inference_mode():
            gen = model.generate(**inp, max_new_tokens=96, do_sample=False)
        return processor.batch_decode(gen[:, inp["input_ids"].shape[1]:],
                                      skip_special_tokens=True)[0].strip()

    with open(pred_path, "a") as f:
        for i, g in enumerate(todo):
            dur = g["duration"]
            starts = []
            t = 0.0
            while t < dur:
                starts.append(t)
                t += args.stride
            present = []
            n_win = 0
            for ws in starts:
                wlen = min(args.window, dur - ws)
                if wlen < 3:
                    continue
                n_win += 1
                clip = TMP / f"{g['query_id']}_{int(ws)}.mp4"
                try:
                    cut(g["video_path"], ws, wlen, clip)
                    reply = ask(str(clip.resolve()),
                                WIN_PROMPT.format(wlen=wlen, query=g["query_text"]))
                finally:
                    clip.unlink(missing_ok=True)
                if "not present" in reply.lower():
                    continue
                loc = parse_time_span(reply)
                if loc:
                    a = max(0.0, min(ws + loc[0], dur))
                    b = max(0.0, min(ws + loc[1], dur))
                    if b > a:
                        present.append([a, b])
            pred, support = merge_cluster(present)
            f.write(json.dumps({"query_id": g["query_id"], "pred": pred,
                                "n_windows": n_win, "n_present": len(present),
                                "support": support}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i+1}/{len(todo)}] {g['query_id']}: {pred} "
                  f"(support {support}/{n_win} windows)", flush=True)


if __name__ == "__main__":
    main()
