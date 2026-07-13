#!/usr/bin/env bash
# Resume of run_caption_gen_pilot.sh: qwen_audio_vl and af3_vl already completed
# successfully (see logs/caption_gen_pilot/{qwen_audio_vl,af3_vl}.log), so this only
# runs the remaining models. secap_qwen previously got killed mid-load (machine's
# GPU-idle reclaim fired because gpu_keepalive.sh had also died) -- rerun it first.
set -u
cd /work/mzha0323/video_query_generation
source env.sh 2>/dev/null

PY=conda_envs/video_env/bin/python
OUT_ROOT=output/caption_gen_pilot
LOG_DIR=logs/caption_gen_pilot
mkdir -p "$LOG_DIR"

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

echo "=== pilot caption gen RESUME $(date) ==="
run_model secap_qwen    --vllm-video
run_model avocado
run_model timechat
run_model qwen3_omni
echo ""; echo "=== pilot caption gen RESUME END $(date) ==="
