#!/usr/bin/env bash
# Sequential batch caption generation: all 6 models x 19 videos (data/pilot_study).
# One model at a time (python exits between models -> GPU freed, no OOM).
# Per-segment disk cache -> resumable. set +e so one model failing doesn't abort the rest.
set -u
cd /work/mzha0323/video_query_generation
source env.sh 2>/dev/null

PY=conda_envs/video_env/bin/python
OUT_ROOT=output/caption_gen_validate
LOG_DIR=logs/caption_gen_validate
mkdir -p "$LOG_DIR"

# qwen3_omni (30B) is slowest; secap failed last time -> run it last so it can't block others.
MODELS=(qwen_audio_vl avocado timechat af3_vl qwen3_omni secap_qwen)

echo "=== batch caption gen START $(date) ==="
for m in "${MODELS[@]}"; do
  echo ""
  echo ">>> [$m] START $(date)"
  set +e
  $PY run_caption_generation.py \
    --caption-model "$m" \
    --video-dir data/pilot_study \
    --output "$OUT_ROOT/$m" \
    > "$LOG_DIR/$m.log" 2>&1
  rc=$?
  set -e
  if [ $rc -eq 0 ]; then
    echo ">>> [$m] DONE ok $(date)"
  else
    echo ">>> [$m] FAILED rc=$rc $(date) (see $LOG_DIR/$m.log)"
  fi
done
echo ""
echo "=== batch caption gen END $(date) ==="
