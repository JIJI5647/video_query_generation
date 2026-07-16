"""Zero-shot temporal grounding with Qwen2.5-VL / Qwen3-VL via transformers.

Official protocol (Qwen blog): free-text prompt, answer like "from X seconds
to Y seconds"; fps is passed through qwen_vl_utils -> processor so mRoPE's
absolute-time clock stays calibrated (the load-bearing detail; vLLM's OpenAI
endpoint does not honor fps, hence transformers here).

  python grounding_baselines/run_qwenvl.py \
      --model Qwen/Qwen2.5-VL-7B-Instruct \
      --gold output/grounding_eval/gold.jsonl \
      --output output/grounding_eval/qwen25vl [--limit 5]
"""
import argparse, json, re
from pathlib import Path

PROMPT = ("Give the query: '{query}', when does the described content occur "
          "in the video? Answer with the start and end time in seconds, "
          "in the format 'from X seconds to Y seconds'.")

MMSS = re.compile(r"\b(\d{1,2}):(\d{2})(?::(\d{2}))?\b")
PAIR = re.compile(r"(\d+(?:\.\d+)?)\s*(?:seconds?)?\s*(?:to|-|–|and)\s*"
                  r"(\d+(?:\.\d+)?)\s*seconds?", re.I)
NUMS = re.compile(r"\d+(?:\.\d+)?")


def parse_time_span(text):
    """Return [start, end] seconds or None. Tries, in order: explicit
    'X to/- Y seconds' pair, mm:ss timestamps, first two bare numbers."""
    m = PAIR.search(text)
    if m:
        s, e = float(m.group(1)), float(m.group(2))
        return [min(s, e), max(s, e)] if s != e else None
    ts = MMSS.findall(text)
    if ts:
        def sec(t):
            h_or_m, m_or_s, s = t
            if s:
                return int(h_or_m) * 3600 + int(m_or_s) * 60 + int(s)
            return int(h_or_m) * 60 + int(m_or_s)
        if len(ts) >= 2:
            a, b = sec(ts[0]), sec(ts[1])
            return [min(a, b), max(a, b)] if a != b else None
        return None  # a single mm:ss point is not a span
    nums = [float(x) for x in NUMS.findall(text)]
    if len(nums) >= 2 and nums[0] != nums[1]:
        return [min(nums[0], nums[1]), max(nums[0], nums[1])]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gold", default="output/grounding_eval/gold.jsonl")
    ap.add_argument("--output", required=True)
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--max-pixels", type=int, default=360 * 420,
                    help="per-frame pixel budget for video frames")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText as Cls
    from qwen_vl_utils import process_vision_info

    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    pred_path = out / "predictions.jsonl"
    done = set()
    if pred_path.exists():
        done = {json.loads(l)["query_id"] for l in open(pred_path) if l.strip()}

    gold = [json.loads(l) for l in open(args.gold) if l.strip()]
    if args.limit:
        gold = gold[:args.limit]
    todo = [g for g in gold if g["query_id"] not in done]
    print(f"{len(todo)} queries to run ({len(done)} cached)", flush=True)
    if not todo:
        return

    model = Cls.from_pretrained(args.model, dtype=torch.bfloat16,
                                attn_implementation="sdpa", device_map="cuda")
    processor = AutoProcessor.from_pretrained(args.model)

    with open(pred_path, "a") as f:
        for i, g in enumerate(todo):
            messages = [{"role": "user", "content": [
                {"type": "video", "video": f"file://{Path(g['video_path']).resolve()}",
                 "fps": args.fps, "max_pixels": args.max_pixels},
                {"type": "text", "text": PROMPT.format(query=g["query_text"])},
            ]}]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            images, videos, video_kwargs = process_vision_info(
                messages, return_video_kwargs=True, return_video_metadata=True)
            if videos is not None:  # split (tensor, metadata) pairs
                videos, metadata = zip(*videos)
                videos, metadata = list(videos), list(metadata)
            else:
                metadata = None
            if i == 0:
                print(f"video tensor: {tuple(videos[0].shape)} "
                      f"kwargs={video_kwargs} metadata={metadata[0]} "
                      f"duration={g['duration']}s", flush=True)
            inputs = processor(text=[text], images=images, videos=videos,
                               video_metadata=metadata,
                               return_tensors="pt", **video_kwargs).to("cuda")
            with torch.inference_mode():
                gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                     do_sample=False)
            reply = processor.batch_decode(
                gen[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True)[0].strip()
            pred = parse_time_span(reply)
            if pred:  # clamp to video duration
                pred = [max(0.0, min(pred[0], g["duration"])),
                        max(0.0, min(pred[1], g["duration"]))]
                if pred[1] <= pred[0]:
                    pred = None
            f.write(json.dumps({"query_id": g["query_id"], "pred": pred,
                                "raw": reply}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i+1}/{len(todo)}] {g['query_id']}: {pred}  <- {reply[:80]!r}",
                  flush=True)


if __name__ == "__main__":
    main()
