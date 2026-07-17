"""Qwen3-VL temporal grounding via a vLLM OpenAI endpoint (fast path).

Same prompt + parsing as run_qwenvl.py, but sends the video to a served model
over HTTP instead of loading transformers in-process. CAVEAT: vLLM's OpenAI
endpoint may not honor per-request fps for Qwen3-VL, which is load-bearing for
mRoPE absolute-time grounding — so this MUST be validated against the
transformers path (run_qwenvl.py) before trusting it. Pass fps via
mm_processor_kwargs; if the served timestamps diverge from transformers, fall
back to run_qwenvl.py.

  python grounding_baselines/run_qwenvl_vllm.py --gold <gold.jsonl> \
      --output <dir> --base-url http://localhost:8000/v1
"""
import argparse, json, os, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from grounding_baselines.run_qwenvl import parse_time_span, PROMPT
from openai import OpenAI


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-30B-A3B-Instruct")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    pred_path = out / "predictions.jsonl"
    done = {json.loads(l)["query_id"] for l in open(pred_path)} if pred_path.exists() else set()
    gold = [json.loads(l) for l in open(args.gold) if l.strip()]
    if args.limit:
        gold = gold[:args.limit]
    todo = [g for g in gold if g["query_id"] not in done]
    print(f"{len(todo)} to run ({len(done)} cached)", flush=True)
    if not todo:
        return

    client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=600)
    lock = threading.Lock()
    f = open(pred_path, "a")
    cnt = [0]

    def run_one(g):
        uri = f"file://{Path(g['video_path']).resolve()}"
        try:
            r = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": [
                    {"type": "video_url", "video_url": {"url": uri}},
                    {"type": "text", "text": PROMPT.format(query=g["query_text"])},
                ]}],
                max_tokens=args.max_tokens, temperature=0.0,
                extra_body={"mm_processor_kwargs": {"fps": args.fps}})
            reply = (r.choices[0].message.content or "").strip()
        except Exception as ex:
            reply = f"__ERROR__ {str(ex)[:150]}"
        pred = parse_time_span(reply) if not reply.startswith("__ERROR__") else None
        if pred:
            pred = [max(0.0, min(pred[0], g["duration"])), max(0.0, min(pred[1], g["duration"]))]
            if pred[1] <= pred[0]:
                pred = None
        with lock:
            cnt[0] += 1
            f.write(json.dumps({"query_id": g["query_id"], "pred": pred, "raw": reply[:120]},
                               ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{cnt[0]}/{len(todo)}] {g['query_id']}: {pred}", flush=True)

    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        list(ex.map(run_one, todo))
    f.close()


if __name__ == "__main__":
    main()
