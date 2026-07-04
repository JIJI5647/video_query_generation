#!/usr/bin/env bash
# Verifier prompt ablation (per-dimension architecture): run each strategy variant
# (p0..p8) over the SAME queries, then score against gold with eval_verification.py.
#
# Each variant is judged as 3 separate inferences (relevance & query_quality from
# query text only; answerability watching the clip), composed from the prompt files
# under prompts/perdim/. Each variant -> its own dir: ${OUT_ROOT}/verify_<variant>/
#
# Run on the GPU server (qwen3_omni verify):
#     bash run_verification_sweep.sh
#     OUT_ROOT=output/run3 QUERIES_DIR=data/test5_eval VIDEO_DIR=data/pilot_study PARALLEL=4 bash run_verification_sweep.sh

set -u

QUERIES_DIR="${QUERIES_DIR:-data/test5_eval}"
VIDEO_DIR="${VIDEO_DIR:-data/pilot_study}"
OUT_ROOT="${OUT_ROOT:-output}"
BACKEND="${BACKEND:-qwen3_omni}"          # qwen3_omni | gemini
VIDEO_READER="${VIDEO_READER:-decord}"
PARALLEL="${PARALLEL:-4}"
VERIFICATION_MODEL="${VERIFICATION_MODEL:-gemini-3.1-flash-lite}"
# Qwen3-Omni verify knobs (BACKEND=qwen3_omni). Set QWEN_MODEL_PATH to the
# Thinking reasoning checkpoint (Qwen/Qwen3-Omni-30B-A3B-Thinking) and bump
# QWEN_MAX_TOKENS (e.g. 8192) so its long CoT isn't truncated before the JSON.
QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-}"
QWEN_MAX_TOKENS="${QWEN_MAX_TOKENS:-}"

QWEN_ARGS=()
[ -n "$QWEN_MODEL_PATH" ] && QWEN_ARGS+=(--qwen-model-path "$QWEN_MODEL_PATH")
[ -n "$QWEN_MAX_TOKENS" ] && QWEN_ARGS+=(--qwen-max-tokens "$QWEN_MAX_TOKENS")

VARIANTS=(
  p0_norule p1_rule p2_role p3_fewshot p4_zscot
  p5_fewshotcot p6_rolefewshot p7_rolecot p8_rawcot
)

mkdir -p logs/verify_sweep

for v in "${VARIANTS[@]}"; do
  out="${OUT_ROOT}/verify_${v}"
  echo "=========================================================="
  echo "[$(date '+%F %T')] verify ${v}  (per-dimension) -> ${out}"
  echo "=========================================================="
  python -u run_verification.py \
    --queries-dir "$QUERIES_DIR" \
    --video-dir "$VIDEO_DIR" \
    --output "$out" \
    --per-dimension --variant "$v" \
    --verify-rewrite-backend "$BACKEND" \
    --verification-model "$VERIFICATION_MODEL" \
    --qwen-video-reader-backend "$VIDEO_READER" \
    ${QWEN_ARGS[@]+"${QWEN_ARGS[@]}"} \
    --parallel "$PARALLEL" \
    > "logs/verify_sweep/${v}.log" 2>&1
  echo "[$(date '+%F %T')] done ${v} (exit $?)  log: logs/verify_sweep/${v}.log"
done

echo "All variants done. Score with:"
echo "  python eval_verification.py --gold ${QUERIES_DIR}/gold.jsonl --results ${OUT_ROOT}/verify_*/verification_results.jsonl --csv ${OUT_ROOT}/verify_metrics.csv"
