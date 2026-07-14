#!/usr/bin/env bash
# Scale-up of experiment ②′ to all 19 videos:
#   v2 emotion events (Gemini reads captions, prompt v2) for 19 videos, then
#   3 generation arms from the SAME events:
#     gemini_text     — Gemini text v12 (production baseline)
#     qwen_watch      — Qwen3-Omni-Instruct watches each event's clip(s)
#     nemotron_watch  — Nemotron-3-Nano-Omni watches each event's clip(s)
#   then verify all 3 with the p7_rolecot Qwen-Instruct vLLM verifier.
# Serve order minimizes swaps: [Instruct up] qwen_watch -> [Nemotron] nemotron_watch
#   -> [Instruct] verify x3. Resumable: skips outputs that already exist.
set -u
cd /work/mzha0323/video_query_generation
source env.sh 2>/dev/null
PY=conda_envs/video_env/bin/python
OUT=output/gen_v2ev19
LOG=logs/gen_v2ev19
mkdir -p "$OUT" "$LOG"

# all 19 pilot video ids from the caption source
VIDS=$($PY -c "
import json
vs=sorted(set(json.loads(l)['video_id'] for l in open('output/eval_unified19/qwen3_omni/raw_captions.jsonl') if l.strip()))
print(','.join(vs))")
echo "videos: $VIDS" | head -c 300; echo

stop_serve() {
  APIPID=$(ps -eo pid,cmd | grep "vllm serve" | grep -v grep | awk '{print $1}' | head -1)
  [ -n "${APIPID:-}" ] && kill -TERM "$APIPID" 2>/dev/null
  pkill -TERM -f "VLLM::EngineCore" 2>/dev/null
  sleep 15
}
wait_endpoint() {  # $1 = grep pattern
  for i in $(seq 1 60); do
    curl -s -m 3 http://localhost:8000/v1/models 2>/dev/null | grep -q "$1" && return 0
    sleep 10
  done
  return 1
}

echo "=== gen_v2ev19 START $(date) ==="

# 1. v2 events for 19 videos (Gemini text, no GPU)
if [ ! -f "$OUT/events/emotion_events.jsonl" ]; then
  echo ">>> [events] START $(date)"
  $PY -u mm_event_pilot.py --backend gemini-text --videos "$VIDS" \
    --output "$OUT/events" > "$LOG/events.log" 2>&1 || { echo "events FAILED"; exit 1; }
  echo ">>> [events] DONE $(date)"
else echo ">>> [events] SKIP"; fi

# 2a. Gemini text arm (no GPU)
if [ ! -f "$OUT/gemini_text/initial_queries.jsonl" ]; then
  echo ">>> [gemini_text] START $(date)"
  $PY -u gen_text_from_events.py --events-dir "$OUT/events" --videos "$VIDS" \
    --output "$OUT/gemini_text" > "$LOG/gemini_text.log" 2>&1 || echo "gemini_text FAILED"
  echo ">>> [gemini_text] DONE $(date)"
else echo ">>> [gemini_text] SKIP"; fi

# 2b. Qwen watch arm (needs Instruct serve — assumed up; start if not)
if [ ! -f "$OUT/qwen_watch/initial_queries.jsonl" ]; then
  if ! curl -s -m 3 http://localhost:8000/v1/models 2>/dev/null | grep -q Instruct; then
    stop_serve
    MODEL="Qwen/Qwen3-Omni-30B-A3B-Instruct" PORT=8000 GPU_UTIL=0.55 \
      nohup bash run_vllm_serve_qwen.sh > "$LOG/serve_qwen1.log" 2>&1 &
    wait_endpoint Instruct || { echo "Instruct serve FAILED"; exit 1; }
  fi
  echo ">>> [qwen_watch] START $(date)"
  $PY -u mm_gen_pilot.py --videos "$VIDS" --events-dir "$OUT/events" \
    --model "Qwen/Qwen3-Omni-30B-A3B-Instruct" --max-tokens 2048 \
    --output "$OUT/qwen_watch" > "$LOG/qwen_watch.log" 2>&1 || echo "qwen_watch FAILED"
  echo ">>> [qwen_watch] DONE $(date)"
else echo ">>> [qwen_watch] SKIP"; fi

# 2c. Nemotron watch arm (swap serve)
if [ ! -f "$OUT/nemotron_watch/initial_queries.jsonl" ]; then
  stop_serve
  MODEL="nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8" PORT=8000 GPU_UTIL=0.55 \
    nohup bash run_vllm_serve.sh > "$LOG/serve_nemotron.log" 2>&1 &
  wait_endpoint Nemotron || { echo "Nemotron serve FAILED"; exit 1; }
  echo ">>> [nemotron_watch] START $(date)"
  $PY -u mm_gen_pilot.py --videos "$VIDS" --events-dir "$OUT/events" \
    --model "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8" --max-tokens 8192 \
    --output "$OUT/nemotron_watch" > "$LOG/nemotron_watch.log" 2>&1 || echo "nemotron_watch FAILED"
  echo ">>> [nemotron_watch] DONE $(date)"
else echo ">>> [nemotron_watch] SKIP"; fi

# 3. Verify all arms on the Instruct serve
stop_serve
MODEL="Qwen/Qwen3-Omni-30B-A3B-Instruct" PORT=8000 GPU_UTIL=0.55 \
  nohup bash run_vllm_serve_qwen.sh > "$LOG/serve_qwen2.log" 2>&1 &
wait_endpoint Instruct || { echo "Instruct serve FAILED"; exit 1; }
for m in gemini_text qwen_watch nemotron_watch; do
  if [ -f "$OUT/$m/verification_results.jsonl" ]; then echo ">>> [verify:$m] SKIP"; continue; fi
  [ -f "$OUT/$m/initial_queries.jsonl" ] || { echo ">>> [verify:$m] no queries — skip"; continue; }
  echo ">>> [verify:$m] START $(date)"
  $PY -u run_verification.py --queries-dir "$OUT/$m" --video-dir data/pilot_study \
    --output "$OUT/$m" --verify-rewrite-backend qwen_omni_vllm \
    --qwen-vllm-base-url http://localhost:8000/v1 \
    --qwen-vllm-model "Qwen/Qwen3-Omni-30B-A3B-Instruct" \
    --per-dimension --variant p7_rolecot --parallel 4 \
    > "$LOG/verify_$m.log" 2>&1 || echo ">>> [verify:$m] FAILED"
  echo ">>> [verify:$m] DONE $(date)"
done

stop_serve
echo "=== gen_v2ev19 END $(date) ==="
