#!/usr/bin/env bash
# Full 2s-segment HYBRID pipeline on ALL 19 videos (cap removed), for the
# 5s-vs-2s comparison at scale. vLLM Qwen3-Omni captioning (served once, reused
# for caption + refine + verify), then Qwen3-VL downstream grounding.
set -u
cd /work/mzha0323/video_query_generation
source env.sh 2>/dev/null
PY=conda_envs/video_env/bin/python
OUT=output/hybrid_2s_19
mkdir -p "$OUT" logs
VIDS=$($PY -c "import json;print(','.join(sorted(set(json.loads(l)['video_id'] for l in open('output/eval_unified19/qwen3_omni/raw_captions.jsonl') if l.strip()))))")

serve_qwen() {
  MODEL="Qwen/Qwen3-Omni-30B-A3B-Instruct" PORT=8000 GPU_UTIL=0.7 MAX_LEN=65536 \
    nohup bash run_vllm_serve_qwen.sh > logs/2s19_serve.log 2>&1 &
  for i in $(seq 1 120); do
    curl -s -m 3 http://localhost:8000/v1/models 2>/dev/null | grep -q Instruct && return 0
    sleep 10
  done
  return 1
}
stop_serve() {
  APIPID=$(ps -eo pid,cmd | grep "[v]llm serve" | awk '{print $1}' | head -1)
  [ -n "${APIPID:-}" ] && kill -TERM "$APIPID"
  pkill -TERM -f "VLLM::[E]ngineCore" 2>/dev/null; sleep 15
}

echo "=== [1/7] serve Qwen3-Omni $(date) ==="
serve_qwen || { echo "serve FAILED"; exit 1; }

echo "=== [2/7] caption @2s via vLLM $(date) ==="
if [ ! -f "$OUT/captions/raw_captions.jsonl" ]; then
  $PY -u run_caption_generation.py --caption-model nemotron_omni --video-dir data/pilot_study \
    --video-ids "$VIDS" --segment-seconds 2 --stride 2 --output "$OUT/captions" \
    --nemotron-model "Qwen/Qwen3-Omni-30B-A3B-Instruct" \
    --nemotron-base-url http://localhost:8000/v1 --nemotron-no-thinking --caption-parallel 8 \
    > logs/2s19_caption.log 2>&1 || { echo "caption FAILED"; stop_serve; exit 1; }
fi
# patch video_id into segments (run_caption_generation omits it)
$PY - <<'PYEOF'
import json
from pathlib import Path
fn="output/hybrid_2s_19/captions/segments.jsonl"
rows=[json.loads(l) for l in open(fn)]
ch=0
for r in rows:
    if not r.get("video_id"):
        p=Path(r["clip_path"]).parts; r["video_id"]=p[p.index("processed_segments")+1]; ch+=1
open(fn,"w").write("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows))
print(f"patched video_id: {ch}/{len(rows)}")
PYEOF

echo "=== [3/7] events (no-cap) + gen $(date) ==="
$PY mm_event_pilot.py --backend gemini-text --videos "$VIDS" --captions-dir "$OUT/captions" \
  --output "$OUT/events" > logs/2s19_events.log 2>&1 || { echo "events FAILED"; stop_serve; exit 1; }
$PY gen_text_from_events.py --events-dir "$OUT/events" --videos "$VIDS" \
  --captions-dir "$OUT/captions" --output "$OUT/gemini_base" > logs/2s19_gen.log 2>&1 \
  || { echo "gen FAILED"; stop_serve; exit 1; }

echo "=== [4/7] refine + guardrail $(date) ==="
$PY hybrid_refine.py --base "$OUT/gemini_base" --model "Qwen/Qwen3-Omni-30B-A3B-Instruct" \
  --thinking none --output "$OUT/refine" > logs/2s19_refine.log 2>&1
$PY hybrid_guardrail.py --in "$OUT/refine" --out "$OUT/final" > logs/2s19_guardrail.log 2>&1

echo "=== [5/7] verify $(date) ==="
$PY run_verification.py --queries-dir "$OUT/final" --video-dir data/pilot_study --output "$OUT/final" \
  --verify-rewrite-backend qwen_omni_vllm --qwen-vllm-base-url http://localhost:8000/v1 \
  --qwen-vllm-model "Qwen/Qwen3-Omni-30B-A3B-Instruct" --per-dimension --variant p7_rolecot \
  --parallel 4 > logs/2s19_verify.log 2>&1
stop_serve

echo "=== [6/7] build grounding input (pass) $(date) ==="
$PY - <<'PYEOF'
import json, subprocess
from pathlib import Path
def load(p): return [json.loads(l) for l in open(p)]
OUT=Path("output/hybrid_2s_19")
ver={x["query_id"]:x for x in load(OUT/"final/verification_results.jsonl")}
durs={}
def dur(v):
    if v in durs: return durs[v]
    d=float(subprocess.run(["ffprobe","-v","error","-show_entries","format=duration","-of","csv=p=0",str(Path("data/pilot_study")/f"{v}.mp4")],capture_output=True,text=True).stdout.strip()); durs[v]=d; return d
g=OUT/"grounding"; g.mkdir(parents=True,exist_ok=True); n=0
with open(g/"gold.jsonl","w") as f:
    for r in load(OUT/"final/initial_queries.jsonl"):
        for q in r["queries"]:
            if ver.get(q["query_id"],{}).get("decision")!="pass": continue
            v=r["video_id"]
            f.write(json.dumps({"query_id":q["query_id"],"video_id":v,"video_path":str(Path("data/pilot_study")/f"{v}.mp4"),"duration":round(dur(v),2),"query_text":q["query_text"],"query_type":q["query_type"],"gold_ranges":[[float(q["time_range"][0]),float(q["time_range"][1])]],"model_time_range":q["time_range"]},ensure_ascii=False)+"\n"); n+=1
print(f"grounding input: {n}")
PYEOF

echo "=== [7/7] Qwen3-VL downstream grounding $(date) ==="
FORCE_QWENVL_VIDEO_READER=decord $PY grounding_baselines/run_qwenvl.py \
  --model Qwen/Qwen3-VL-30B-A3B-Instruct --gold "$OUT/grounding/gold.jsonl" \
  --output "$OUT/grounding/qwen3vl_30b" > logs/2s19_grounding.log 2>&1

echo "=== 2s@19 COMPLETE $(date) ==="
