#!/usr/bin/env bash
# Exp3 (rest): the 3 remaining caption models under the SAME new unified pipeline
# as run_exp3_unified.sh (unstructured visual+audio captions, length limits,
# event-merge guard, af3 no-emotion audio fix, emotion-leak neutralizer).
# Same 5 test videos -> directly comparable to qwen3_omni/qwen_audio_vl + v12.
# Two-stage per model, one at a time (GPU freed between models -> no OOM).
set -u
cd /work/mzha0323/video_query_generation
source env.sh 2>/dev/null

PY=conda_envs/video_env/bin/python
VIDS="emostim_02_WhenHarryMetSally_clip_1,emostim_03_MrBeansHoliday_clip_4,emostim_06_TheChamp_clip_3,meld_01_dia337,meld_03_dia572"
VIDEO_DIR=data/pilot_study
CAP_ROOT=output/exp3_unified/captions
EVAL_ROOT=output/exp3_unified/eval
LOG_DIR=logs/exp3_unified
mkdir -p "$LOG_DIR" "$CAP_ROOT" "$EVAL_ROOT"

MODELS=(timechat avocado af3_vl)   # af3_vl last (subprocess env, slowest to set up)

echo "=== Exp3 REST (3 models) START $(date) ==="
for m in "${MODELS[@]}"; do
  echo ""
  echo ">>> [$m] CAPTION START $(date)"
  set +e
  $PY -u run_caption_generation.py \
    --caption-model "$m" \
    --video-dir "$VIDEO_DIR" \
    --video-ids "$VIDS" \
    --output "$CAP_ROOT/$m" \
    > "$LOG_DIR/${m}_caption.log" 2>&1
  rc=$?
  set -e
  if [ $rc -ne 0 ]; then
    echo ">>> [$m] CAPTION FAILED rc=$rc $(date) (see $LOG_DIR/${m}_caption.log)"
    continue
  fi
  echo ">>> [$m] CAPTION DONE $(date); EVAL START"
  set +e
  $PY -u rerun_generation.py \
    --captions-dir "$CAP_ROOT/$m" \
    --video-dir "$VIDEO_DIR" \
    --output "$EVAL_ROOT/$m" \
    --parallel 4 \
    > "$LOG_DIR/${m}_eval.log" 2>&1
  rc=$?
  set -e
  if [ $rc -eq 0 ]; then
    echo ">>> [$m] EVAL DONE ok $(date)"
  else
    echo ">>> [$m] EVAL FAILED rc=$rc $(date) (see $LOG_DIR/${m}_eval.log)"
  fi
done
echo ""
echo "=== Exp3 REST END $(date) ==="
