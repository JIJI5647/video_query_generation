#!/usr/bin/env bash
# Full pipeline at 2-SECOND segments on 3 videos, to compare vs the 5s run.
# caption(2s) -> events(no-cap) -> gen(disambig) -> Qwen3-Omni ground -> guardrail
# -> verify -> Qwen3-VL downstream grounding.
set -u
cd /work/mzha0323/video_query_generation
source env.sh 2>/dev/null
PY=conda_envs/video_env/bin/python
VIDS="emostim_06_TheChamp_clip_3,emostim_01_TheShining_clip_2,meld_03_dia572"
OUT=output/cap_test_2s
mkdir -p "$OUT" logs

echo "=== [1/6] caption @2s (qwen3_omni) $(date) ==="
$PY -u run_caption_generation.py --caption-model qwen3_omni --video-dir data/pilot_study \
  --video-ids "$VIDS" --segment-seconds 2 --stride 2 --output "$OUT/captions" \
  > logs/2s_caption.log 2>&1 || { echo "caption FAILED"; exit 1; }

echo "=== [2/6] events (no-cap) $(date) ==="
$PY mm_event_pilot.py --backend gemini-text --videos "$VIDS" --captions-dir "$OUT/captions" \
  --output "$OUT/events" > logs/2s_events.log 2>&1 || { echo "events FAILED"; exit 1; }

echo "=== [3/6] generation (disambig) $(date) ==="
$PY gen_text_from_events.py --events-dir "$OUT/events" --videos "$VIDS" \
  --captions-dir "$OUT/captions" --output "$OUT/gemini_base" > logs/2s_gen.log 2>&1 \
  || { echo "gen FAILED"; exit 1; }

echo "=== [4/6] serve Qwen3-Omni + refine/guardrail/verify $(date) ==="
MODEL="Qwen/Qwen3-Omni-30B-A3B-Instruct" PORT=8000 GPU_UTIL=0.7 MAX_LEN=65536 \
  nohup bash run_vllm_serve_qwen.sh > logs/2s_serve.log 2>&1 &
for i in $(seq 1 90); do
  curl -s -m 3 http://localhost:8000/v1/models 2>/dev/null | grep -q Instruct && break
  sleep 10
done
$PY hybrid_refine.py --base "$OUT/gemini_base" --model "Qwen/Qwen3-Omni-30B-A3B-Instruct" \
  --thinking none --output "$OUT/refine" > logs/2s_refine.log 2>&1
$PY hybrid_guardrail.py --in "$OUT/refine" --out "$OUT/final" > logs/2s_guardrail.log 2>&1
$PY run_verification.py --queries-dir "$OUT/final" --video-dir data/pilot_study --output "$OUT/final" \
  --verify-rewrite-backend qwen_omni_vllm --qwen-vllm-base-url http://localhost:8000/v1 \
  --qwen-vllm-model "Qwen/Qwen3-Omni-30B-A3B-Instruct" --per-dimension --variant p7_rolecot \
  --parallel 4 > logs/2s_verify.log 2>&1
APIPID=$(ps -eo pid,cmd | grep "[v]llm serve" | awk '{print $1}' | head -1)
[ -n "${APIPID:-}" ] && kill -TERM "$APIPID"; pkill -TERM -f "VLLM::[E]ngineCore" 2>/dev/null; sleep 15

echo "=== [5/6] build grounding input (pass queries) $(date) ==="
$PY - <<'PYEOF'
import json, subprocess
from pathlib import Path
def load(p): return [json.loads(l) for l in open(p)]
OUT=Path("output/cap_test_2s")
ver={x["query_id"]:x for x in load(OUT/"final/verification_results.jsonl")}
fin=load(OUT/"final/initial_queries.jsonl")
durs={}
def dur(v):
    if v in durs: return durs[v]
    d=float(subprocess.run(["ffprobe","-v","error","-show_entries","format=duration","-of","csv=p=0",str(Path("data/pilot_study")/f"{v}.mp4")],capture_output=True,text=True).stdout.strip()); durs[v]=d; return d
g=OUT/"grounding"; g.mkdir(parents=True,exist_ok=True); n=0
with open(g/"gold.jsonl","w") as f:
    for r in fin:
        for q in r["queries"]:
            if ver.get(q["query_id"],{}).get("decision")!="pass": continue
            v=r["video_id"]
            f.write(json.dumps({"query_id":q["query_id"],"video_id":v,"video_path":str(Path("data/pilot_study")/f"{v}.mp4"),"duration":round(dur(v),2),"query_text":q["query_text"],"query_type":q["query_type"],"gold_ranges":[[float(q["time_range"][0]),float(q["time_range"][1])]],"model_time_range":q["time_range"]},ensure_ascii=False)+"\n"); n+=1
print(f"grounding input: {n} pass queries")
PYEOF

echo "=== [6/6] Qwen3-VL downstream grounding $(date) ==="
FORCE_QWENVL_VIDEO_READER=decord $PY grounding_baselines/run_qwenvl.py \
  --model Qwen/Qwen3-VL-30B-A3B-Instruct --gold "$OUT/grounding/gold.jsonl" \
  --output "$OUT/grounding/qwen3vl_30b" > logs/2s_grounding.log 2>&1

echo "=== 2S PIPELINE COMPLETE $(date) ==="
