#!/usr/bin/env bash
# Downstream eval for the 6-model x 19-video unified captions: for each caption
# model, generation (v12) + regrounding OFF + verify (Qwen3-Omni p7_rolecot via
# the vLLM OpenAI endpoint). Starts one shared Qwen3-Omni verify serve, runs all
# 6 models against it, stops it. Resumable: skips a model whose final_queries exists.
set -u
cd /work/mzha0323/video_query_generation
source env.sh 2>/dev/null
PY=conda_envs/video_env/bin/python
CAPS_ROOT=output/caption_gen_unified19
OUT=output/eval_unified19
LOG=logs/eval_unified19
mkdir -p "$OUT" "$LOG"
MODELS=(qwen_audio_vl af3_vl avocado timechat qwen3_omni nemotron_omni)

echo "=== eval_unified19 START $(date) ==="
# 1. Start the Qwen3-Omni verify serve (GPU_UTIL 0.55 ~78GB; ~52GB orphan leaves ~91GB free).
MODEL="Qwen/Qwen3-Omni-30B-A3B-Instruct" PORT=8000 GPU_UTIL=0.55 \
  bash run_vllm_serve_qwen.sh > "$LOG/verify_serve.log" 2>&1 &
SERVE_PID=$!
echo "[eval] verify serve pid=$SERVE_PID, waiting for endpoint..."
ready=0
for i in $(seq 1 90); do
  if curl -s -m 3 http://localhost:8000/v1/models 2>/dev/null | grep -q Qwen3-Omni; then ready=1; break; fi
  if ! kill -0 "$SERVE_PID" 2>/dev/null; then echo "[eval] serve died (see $LOG/verify_serve.log)"; break; fi
  sleep 10
done
if [ "$ready" != "1" ]; then echo ">>> VERIFY SERVE NOT READY — abort"; kill "$SERVE_PID" 2>/dev/null; exit 1; fi
echo "[eval] endpoint ready after ~$((i*10))s"

# 2. Downstream per model (generation v12 + no regrounding + vllm verify).
for m in "${MODELS[@]}"; do
  dir="$OUT/$m"
  if [ -f "$dir/final_queries.jsonl" ]; then echo ">>> [$m] SKIP (exists)"; continue; fi
  [ -f "$CAPS_ROOT/$m/raw_captions.jsonl" ] || { echo ">>> [$m] no captions — skip"; continue; }
  echo ">>> [$m] START $(date)"
  set +e
  $PY -u rerun_generation.py \
    --captions-dir "$CAPS_ROOT/$m" --video-dir data/pilot_study --output "$dir" \
    --no-regrounding \
    --verify-rewrite-backend qwen_omni_vllm \
    --qwen-vllm-base-url http://localhost:8000/v1 \
    --qwen-vllm-model "Qwen/Qwen3-Omni-30B-A3B-Instruct" \
    --parallel 4 > "$LOG/$m.log" 2>&1
  rc=$?; set -e
  [ $rc -eq 0 ] && echo ">>> [$m] DONE $(date)" || echo ">>> [$m] FAILED rc=$rc (see $LOG/$m.log)"
done

# 3. Stop the serve.
kill "$SERVE_PID" 2>/dev/null; pkill -f "vllm serve.*Qwen3-Omni" 2>/dev/null
echo "=== eval_unified19 END $(date) ==="
