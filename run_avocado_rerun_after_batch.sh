#!/usr/bin/env bash
# Re-run avocado's generation+evaluation AFTER the main eval batch finishes.
# avocado's first run drew a one-off Gemini runaway (57 captions -> 180 events ->
# 91 queries, ~10h projected) and was killed; the emotion-event stage now has a
# runaway guard (emotion_events.py: re-sample when events > 2x captions). This
# waits for the current orchestrator (run_eval_pilot_p7_rolecot.sh, PID passed as
# $1) to exit so only ONE model is on the GPU at a time, then re-runs avocado into
# the same output tree with the guarded code.
set -u
cd /work/mzha0323/video_query_generation
source env.sh 2>/dev/null

ORCH_PID="${1:-}"
PY=conda_envs/video_env/bin/python
OUT=output/eval_pilot_p7_rolecot/avocado
LOG=logs/eval_pilot_p7_rolecot/avocado_rerun.log

echo "[avocado-rerun] waiting for main batch (orchestrator pid $ORCH_PID) to finish..."
if [ -n "$ORCH_PID" ]; then
  while kill -0 "$ORCH_PID" 2>/dev/null; do sleep 30; done
fi
# Belt-and-suspenders: also wait until no rerun_generation.py process is on the GPU.
while pgrep -f "rerun_generation.py" >/dev/null 2>&1; do sleep 30; done

echo "[avocado-rerun] main batch done, starting avocado re-run $(date)"
rm -rf "$OUT"
$PY -u rerun_generation.py \
  --captions-dir output/caption_gen_pilot/avocado \
  --video-dir data/pilot_study \
  --output "$OUT" \
  --parallel 4 \
  > "$LOG" 2>&1
rc=$?
[ $rc -eq 0 ] && echo "[avocado-rerun] DONE ok $(date)" \
             || echo "[avocado-rerun] FAILED rc=$rc $(date) (see $LOG)"
