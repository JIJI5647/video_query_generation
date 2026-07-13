#!/usr/bin/env bash
# Generation + evaluation over the 5 completed caption-model outputs in
# output/caption_gen_pilot/ (secap_qwen excluded -- never produced captions).
# Uses rerun_generation.py: Gemini for emotion-events/query-generation, and the
# now-default qwen3_omni verify/rewrite backend (p7_rolecot per-dimension
# prompts, role + CoT). One model at a time (Qwen3-Omni-30B loaded/unloaded per
# process -> no GPU OOM). set +e so one model failing doesn't abort the rest.
#
# --parallel 4: first attempt (verify_parallel=1, default) took ~25s PER
# dimension-call with zero batching (each of relevance/answerability/
# query_quality is its own single-item generate() call) -> ~1hr+ projected per
# video on some videos, days for the full 5-model x 19-video batch. Restarted
# with real batching (relevance/query_quality don't need video so batch cheaply;
# answerability attaches a clip per item, watch GPU memory). `-u` for
# unbuffered stdout so log tails reflect real-time progress (buffered stdout
# made the first attempt's progress invisible for 10+ min at a time).
set -u
cd /work/mzha0323/video_query_generation
source env.sh 2>/dev/null

PY=conda_envs/video_env/bin/python
CAPTIONS_ROOT=output/caption_gen_pilot
OUT_ROOT=output/eval_pilot_p7_rolecot
LOG_DIR=logs/eval_pilot_p7_rolecot
mkdir -p "$LOG_DIR"

MODELS=(af3_vl avocado timechat qwen3_omni qwen_audio_vl)

echo "=== eval (p7_rolecot) batch START $(date) ==="
for m in "${MODELS[@]}"; do
  echo ""
  echo ">>> [$m] START $(date)"
  set +e
  $PY -u rerun_generation.py \
    --captions-dir "$CAPTIONS_ROOT/$m" \
    --video-dir data/pilot_study \
    --output "$OUT_ROOT/$m" \
    --parallel 4 \
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
echo "=== eval (p7_rolecot) batch END $(date) ==="
