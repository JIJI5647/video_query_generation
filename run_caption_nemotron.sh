#!/usr/bin/env bash
# nemotron_omni caption: served model, NOT in-process. Start the Nemotron-3-Nano-
# Omni vLLM OpenAI server (fused_moe_90 kernel already compiled+cached), wait for
# the endpoint, caption all 19 videos over HTTP, then stop the server. Run this
# ONLY when the GPU is free (the served model needs ~40-60GB) — i.e. after the
# in-process caption models finish. Resumable via the per-segment cache.
set -u
cd /work/mzha0323/video_query_generation
source env.sh 2>/dev/null
PY=conda_envs/video_env/bin/python
OUT=output/caption_gen_unified19
LOG=logs/caption_gen_unified19
mkdir -p "$OUT" "$LOG"

MODEL="nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8"
echo "=== nemotron_omni caption START $(date) ==="
# 1. Serve Nemotron-Omni (background). run_vllm_serve.sh handles the ninja/CUDA env.
# GPU_UTIL 0.55 (~78GB): a ~52GB orphan CUDA context (unkillable, other namespace)
# leaves only ~91GB free; 0.85 default (121GB) would OOM.
MODEL="$MODEL" PORT=8000 GPU_UTIL=0.55 \
  bash run_vllm_serve.sh > "$LOG/nemotron_serve.log" 2>&1 &
SERVE_PID=$!
echo "[nemotron] serve pid=$SERVE_PID, waiting for endpoint..."

# 2. Wait for the OpenAI endpoint to come up (up to ~12 min: model load + warmup).
ready=0
for i in $(seq 1 72); do
  if curl -s -m 3 http://localhost:8000/v1/models 2>/dev/null | grep -q Nemotron; then ready=1; break; fi
  if ! kill -0 "$SERVE_PID" 2>/dev/null; then echo "[nemotron] serve died early (see $LOG/nemotron_serve.log)"; break; fi
  sleep 10
done
if [ "$ready" != "1" ]; then
  echo ">>> [nemotron_omni] SERVE NOT READY — abort (see $LOG/nemotron_serve.log)"
  kill "$SERVE_PID" 2>/dev/null
  exit 1
fi
echo "[nemotron] endpoint ready after ~$((i*10))s"

# 3. Caption all 19 videos over HTTP (concurrency-safe served session).
set +e
$PY -u run_caption_generation.py --caption-model nemotron_omni \
  --nemotron-base-url http://localhost:8000/v1 \
  --nemotron-model "$MODEL" \
  --video-dir data/pilot_study --output "$OUT/nemotron_omni" \
  --caption-parallel 4 > "$LOG/nemotron_omni.log" 2>&1
rc=$?
set -e

# 4. Stop the server.
kill "$SERVE_PID" 2>/dev/null
pkill -f "vllm serve.*Nemotron" 2>/dev/null
echo ">>> [nemotron_omni] $([ $rc -eq 0 ] && echo DONE || echo "FAILED rc=$rc") $(date)"
echo "=== nemotron_omni caption END $(date) ==="
