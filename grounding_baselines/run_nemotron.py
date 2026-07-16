"""Zero-shot temporal grounding with Nemotron-3-Nano-Omni via a vLLM
OpenAI-compatible endpoint (audio-visual: the only baseline that hears).

Requires the Nemotron vLLM serve to be up (run_vllm_serve.sh). Videos are
passed as file:// URIs with use_audio_in_video, same transport as
mm_gen_pilot.py. Known risk: Nemotron returned empty responses on long
videos before (context limit) — empty/unparseable replies are recorded as
pred=null and count as parse failures downstream.

  python grounding_baselines/run_nemotron.py \
      --gold output/grounding_eval/gold.jsonl \
      --output output/grounding_eval/nemotron [--limit 5]
"""
import argparse, json, sys, os, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from grounding_baselines.run_qwenvl import parse_time_span
from grounding_baselines.prompts import PARSE_MODE, build_prompt, parse_segments

from openai import OpenAI


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="output/grounding_eval/gold.jsonl")
    ap.add_argument("--output", required=True)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--thinking", choices=["true", "false", "none"], default="true",
                    help="'none' omits chat_template_kwargs entirely (models "
                         "without a thinking mode, e.g. Qwen3-Omni Instruct)")
    ap.add_argument("--no-thinking", dest="thinking", action="store_const",
                    const="false")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--prompt", default="p0", choices=list(PARSE_MODE))
    ap.add_argument("--parallel", type=int, default=1)
    args = ap.parse_args()

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

    client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=600)
    extra_body = {"mm_processor_kwargs": {"use_audio_in_video": True}}
    if args.thinking != "none":
        extra_body["chat_template_kwargs"] = {
            "enable_thinking": args.thinking == "true"}

    lock = threading.Lock()
    counter = [0]

    def run_one(g):
        uri = f"file://{Path(g['video_path']).resolve()}"
        prompt = build_prompt(args.prompt, g["query_text"], g["duration"])
        try:
            resp = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": [
                    {"type": "video_url", "video_url": {"url": uri}},
                    {"type": "text", "text": prompt},
                ]}],
                max_tokens=args.max_tokens, temperature=0.0,
                extra_body=extra_body)
            reply = (resp.choices[0].message.content or "").strip()
        except Exception as ex:
            reply = f"__ERROR__ {str(ex)[:200]}"
        # strip a thinking block if present
        vis = reply.split("</think>")[-1].strip() if "</think>" in reply else reply
        if vis.startswith("__ERROR__"):
            pred = None
        elif PARSE_MODE[args.prompt] == "segments":
            pred = parse_segments(vis, g["duration"])
        else:
            pred = parse_time_span(vis)
        if pred:
            pred = [max(0.0, min(pred[0], g["duration"])),
                    max(0.0, min(pred[1], g["duration"]))]
            if pred[1] <= pred[0]:
                pred = None
        with lock:
            counter[0] += 1
            f.write(json.dumps({"query_id": g["query_id"], "pred": pred,
                                "raw": vis[-300:]}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{counter[0]}/{len(todo)}] {g['query_id']}: {pred}  "
                  f"<- {vis[-80:]!r}", flush=True)

    with open(pred_path, "a") as f:
        if args.parallel > 1:
            with ThreadPoolExecutor(max_workers=args.parallel) as ex:
                list(ex.map(run_one, todo))
        else:
            for g in todo:
                run_one(g)


if __name__ == "__main__":
    main()
