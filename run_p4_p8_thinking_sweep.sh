#!/usr/bin/env bash
# Verifier prompt ablation, Qwen3-Omni-Thinking checkpoint, variants p4-p8 only.
# p0-p3 already done (see docs/progress_log.md); this finishes the 9-variant sweep.
#
#     nohup bash run_p4_p8_thinking_sweep.sh > logs/p4_p8_thinking_sweep.log 2>&1 &
#     disown

set -u

source env.sh
PYTHON="$(pwd)/conda_envs/video_env/bin/python"

QUERIES_DIR="data/test5_eval"
VIDEO_DIR="data/pilot_study"
OUT_ROOT="output"
BACKEND="qwen3_omni"
VIDEO_READER="decord"
PARALLEL=1
QWEN_MODEL_PATH="Qwen/Qwen3-Omni-30B-A3B-Thinking"
QWEN_MAX_TOKENS=8192

VARIANTS=(p4_zscot p5_fewshotcot p6_rolefewshot p7_rolecot p8_rawcot)

mkdir -p logs/verify_sweep

for v in "${VARIANTS[@]}"; do
  out="${OUT_ROOT}/verify_${v}"
  echo "=========================================================="
  echo "[$(date '+%F %T')] verify ${v}  (per-dimension, Thinking) -> ${out}"
  echo "=========================================================="
  "$PYTHON" -u run_verification.py \
    --queries-dir "$QUERIES_DIR" \
    --video-dir "$VIDEO_DIR" \
    --output "$out" \
    --per-dimension --variant "$v" \
    --verify-rewrite-backend "$BACKEND" \
    --qwen-model-path "$QWEN_MODEL_PATH" \
    --qwen-max-tokens "$QWEN_MAX_TOKENS" \
    --qwen-video-reader-backend "$VIDEO_READER" \
    --parallel "$PARALLEL" \
    > "logs/verify_sweep/${v}.log" 2>&1
  echo "[$(date '+%F %T')] done ${v} (exit $?)  log: logs/verify_sweep/${v}.log"
done

echo "p4-p8 thinking sweep (parallel=1) finished."
echo "Score with:"
echo "  python eval_verification.py --gold ${QUERIES_DIR}/gold.jsonl --results ${OUT_ROOT}/verify_p{0,1,2,3,4,5,6,7,8}_*/verification_results.jsonl --csv ${OUT_ROOT}/verify_metrics_thinking.csv"
