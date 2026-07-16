#!/usr/bin/env bash
set -u
cd /work/mzha0323/video_query_generation
source env.sh 2>/dev/null
PY=conda_envs/video_env/bin/python
# swap serve: stop whatever is up, start Nemotron
APIPID=$(ps -eo pid,cmd | grep "[v]llm serve" | awk '{print $1}' | head -1)
[ -n "${APIPID:-}" ] && kill -TERM "$APIPID"
pkill -TERM -f "VLLM::[E]ngineCore" 2>/dev/null
sleep 20
MODEL="nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8" PORT=8000 GPU_UTIL=0.55 MAX_LEN=131072 \
  nohup bash run_vllm_serve.sh > logs/grounding_eval/serve_nemotron_phaseB.log 2>&1 &
for i in $(seq 1 90); do
  curl -s -m 3 http://localhost:8000/v1/models 2>/dev/null | grep -q Nemotron && break
  sleep 10
done
for P in p0 p1_strict p2_cot p3_emotion p4_grid; do
  OUT=output/grounding_eval/sweep_nemotron_$P
  echo ">>> [$P] START $(date)"
  $PY grounding_baselines/run_nemotron.py \
    --model "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8" --thinking false \
    --gold output/grounding_eval/pool_dev.jsonl --output "$OUT" --prompt "$P" --parallel 4 \
    > "logs/grounding_eval/sweep_nemotron_$P.log" 2>&1 || echo "[$P] FAILED"
  echo ">>> [$P] DONE $(date)"
done
echo "PHASE B COMPLETE $(date)"
