#!/usr/bin/env bash
# Ablation: is the emotion-event stage useful?
#   Arm A (exists)  = output/eval_unified19/qwen3_omni  (caption -> events -> gen v12 -> verify)
#   Arm B (this run) = caption -> generation-DIRECT (no event signal) -> verify
# Same cached qwen3_omni captions (19 videos), same verify backend (Qwen3-Omni vLLM
# endpoint, p7_rolecot per-dim), same --no-regrounding. Only difference: the
# generation prompt has NO {events_json} placeholder, so the emotion-event signal
# never reaches generation (events still run inside rerun_generation.py but are
# ignored — isolates the SIGNAL, which is the question).
# Prompt is swapped in and restored on exit (same pattern as run_prompt_sweep.sh).
set -u
cd /work/mzha0323/video_query_generation
source env.sh 2>/dev/null
PY=conda_envs/video_env/bin/python
CAPS=output/caption_gen_unified19/qwen3_omni
OUT=output/ablation_noevent/qwen3_omni
LOG=logs/ablation_noevent
mkdir -p "$OUT" "$LOG"

GEN=prompts/generation_prompt.txt
BACKUP="$LOG/generation_prompt.orig.txt"
cp "$GEN" "$BACKUP"
restore(){ cp "$BACKUP" "$GEN"; echo "[ablation] restored generation_prompt.txt"; }
trap restore EXIT
cp prompts/generation_prompt_direct_noevent.txt "$GEN"
echo "[ablation] swapped in: $(head -1 "$GEN")"

# 1. Serve the Qwen3-Omni verifier (GPU_UTIL 0.55 — ~52GB orphan leaves ~90GB free).
MODEL="Qwen/Qwen3-Omni-30B-A3B-Instruct" PORT=8000 GPU_UTIL=0.55 \
  bash run_vllm_serve_qwen.sh > "$LOG/verify_serve.log" 2>&1 &
SERVE_PID=$!
echo "[ablation] verify serve pid=$SERVE_PID, waiting for endpoint..."
ready=0
for i in $(seq 1 90); do
  if curl -s -m 3 http://localhost:8000/v1/models 2>/dev/null | grep -q Qwen3-Omni; then ready=1; break; fi
  if ! kill -0 "$SERVE_PID" 2>/dev/null; then echo "[ablation] serve died (see $LOG/verify_serve.log)"; break; fi
  sleep 10
done
if [ "$ready" != "1" ]; then echo ">>> SERVE NOT READY — abort"; kill "$SERVE_PID" 2>/dev/null; exit 1; fi
echo "[ablation] endpoint ready after ~$((i*10))s"

# 2. Arm B: generation-direct + verify (identical flags to the eval_unified19 run).
set +e
$PY -u rerun_generation.py \
  --captions-dir "$CAPS" --video-dir data/pilot_study --output "$OUT" \
  --no-regrounding \
  --verify-rewrite-backend qwen_omni_vllm \
  --qwen-vllm-base-url http://localhost:8000/v1 \
  --qwen-vllm-model "Qwen/Qwen3-Omni-30B-A3B-Instruct" \
  --parallel 4 > "$LOG/armB.log" 2>&1
rc=$?
set -e

# 3. Stop the serve.
kill "$SERVE_PID" 2>/dev/null; pkill -f "vllm serve.*Qwen3-Omni" 2>/dev/null
echo ">>> [armB] $([ $rc -eq 0 ] && echo DONE || echo "FAILED rc=$rc") $(date)"
echo "=== ablation_noevent END $(date) ==="
