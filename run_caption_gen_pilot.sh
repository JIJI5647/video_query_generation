#!/usr/bin/env bash
# Pilot caption generation over data/pilot_study (19 videos).
# The 3 Qwen3-VL models run their VIDEO half under vLLM (--vllm-video, ~10-70x faster,
# validated 2026-07-06); the audio halves keep their existing backends. The 3 AV-only
# models (qwen3_omni / avocado / timechat) run the original transformers path for now.
# Sequential (one model-run on the GPU at a time -> no OOM). set +e so one failure
# doesn't abort the rest. Per-segment disk cache -> resumable.
set -u
cd /work/mzha0323/video_query_generation
source env.sh 2>/dev/null

PY=conda_envs/video_env/bin/python
OUT_ROOT=output/caption_gen_pilot
LOG_DIR=logs/caption_gen_pilot
mkdir -p "$LOG_DIR"

# model : extra flags
run_model() {
  local m="$1"; shift
  echo ""; echo ">>> [$m] START $(date)"
  set +e
  $PY run_caption_generation.py \
    --caption-model "$m" \
    --video-dir data/pilot_study \
    --output "$OUT_ROOT/$m" \
    "$@" \
    > "$LOG_DIR/$m.log" 2>&1
  local rc=$?
  set -e
  [ $rc -eq 0 ] && echo ">>> [$m] DONE ok $(date)" \
                || echo ">>> [$m] FAILED rc=$rc $(date) (see $LOG_DIR/$m.log)"
}

echo "=== pilot caption gen START $(date) ==="
# vLLM-accelerated video half:
run_model qwen_audio_vl --vllm-video
run_model af3_vl        --vllm-video
run_model secap_qwen    --vllm-video
# original transformers AV path:
run_model avocado
run_model timechat
run_model qwen3_omni
echo ""; echo "=== pilot caption gen END $(date) ==="
