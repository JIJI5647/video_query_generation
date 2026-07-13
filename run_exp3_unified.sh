#!/usr/bin/env bash
# Exp3: validate the NEW unified caption pipeline (unstructured visual+audio,
# length-limited captions, event-merge guard) end-to-end on real GPU.
# Same 5 test videos as Exp2 (output/exp2_v12) so metrics are directly comparable.
# Two-stage per model, one model at a time (Qwen3-Omni loaded/unloaded per
# process -> no OOM):
#   1. run_caption_generation.py  -> fresh UNIFIED captions
#   2. rerun_generation.py        -> Gemini emotion-events(+merge)+query-gen(v12)
#                                    + qwen3_omni p7_rolecot per-dim verify
# set +e so one model failing doesn't abort the rest.
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

MODELS=(qwen3_omni qwen_audio_vl)

echo "=== Exp3 unified pipeline START $(date) ==="
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
echo "=== Exp3 unified pipeline END $(date) ==="
