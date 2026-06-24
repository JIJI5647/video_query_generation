#!/usr/bin/env bash
# Batch-caption-size quality sweep (transcript OFF).
#
# Fixed: --parallel 4, 5 videos (same set via --seed), no transcript.
# Varies: --caption-batch-size over 1, 2, 4, 8.
# Each batch size gets its OWN output dir + log folder so caption caches and
# logs never mix — compare captions across batch sizes afterwards.
#
# Run on the GPU server (needs GEMINI_API_KEY exported for generation/verify):
#     export GEMINI_API_KEY=...        # required
#     nohup bash run_batch_sweep.sh > logs/batch_sweep/sweep.log 2>&1 &
#     tail -f logs/batch_sweep/sweep.log

set -u

VIDEO_DIR="data/pilot_study"
NUM_VIDEOS=5
SEED=42
PARALLEL=2
BATCH_SIZES=(1 2 4 8)

mkdir -p logs/batch_sweep

for BS in "${BATCH_SIZES[@]}"; do
  OUT="output/batch_sweep/bs${BS}"
  LOGDIR="logs/batch_sweep/bs${BS}"
  mkdir -p "$LOGDIR"
  echo "=========================================================="
  echo "[$(date '+%F %T')] START caption-batch-size=${BS}"
  echo "  output: ${OUT}"
  echo "  log:    ${LOGDIR}/run.log"
  echo "=========================================================="
  python -u run_pipeline.py \
    --video-dir "$VIDEO_DIR" \
    --num-videos "$NUM_VIDEOS" \
    --seed "$SEED" \
    --output "$OUT" \
    --no-transcript \
    --parallel "$PARALLEL" \
    --caption-batch-size "$BS" \
    > "${LOGDIR}/run.log" 2>&1
  echo "[$(date '+%F %T')] DONE caption-batch-size=${BS} (exit $?)"
done

echo "All batch sizes finished."
