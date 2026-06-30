#!/usr/bin/env bash
# Verifier prompt ablation: run each strategy variant (p0..p8) over the SAME
# queries, then score against gold with eval_verification.py.
#
# Default MODE=perdim — the new per-dimension architecture: each variant is judged
# as 3 separate inferences (relevance & query_quality from query text only;
# answerability watching the clip), composed from vdim_template.txt + strategy
# fragments. MODE=combined runs the old single-call combined prompts instead.
#
# Each variant -> its own dir: ${OUT_ROOT}/verify_<variant>/
#
# Run on the GPU server (qwen3_omni verify):
#     bash run_verification_sweep.sh
#     QUERIES_DIR=data/test5_eval VIDEO_DIR=data/pilot_study PARALLEL=4 bash run_verification_sweep.sh
#     MODE=combined bash run_verification_sweep.sh    # old single-call prompts

set -u

QUERIES_DIR="${QUERIES_DIR:-data/test5_eval}"
VIDEO_DIR="${VIDEO_DIR:-data/pilot_study}"
OUT_ROOT="${OUT_ROOT:-output}"
BACKEND="${BACKEND:-qwen3_omni}"          # qwen3_omni | gemini
VIDEO_READER="${VIDEO_READER:-decord}"
PARALLEL="${PARALLEL:-4}"
VERIFICATION_MODEL="${VERIFICATION_MODEL:-gemini-3.1-flash-lite}"
MODE="${MODE:-perdim}"                     # perdim | combined

VARIANTS=(
  p0_norule p1_rule p2_role p3_fewshot p4_zscot
  p5_fewshotcot p6_rolefewshot p7_rolecot p8_rawcot
)

# variant -> combined prompt file (only used when MODE=combined).
combined_file() {
  case "$1" in
    p1_rule) echo "verification_prompt.txt" ;;
    *)       echo "verification_prompt_${1}.txt" ;;
  esac
}

mkdir -p logs/verify_sweep

for v in "${VARIANTS[@]}"; do
  out="${OUT_ROOT}/verify_${v}"
  if [ "$MODE" = "combined" ]; then
    MODE_ARGS=(--using-prompt "prompts/$(combined_file "$v")")
    desc="combined prompt $(combined_file "$v")"
  else
    MODE_ARGS=(--per-dimension --variant "$v")
    desc="per-dimension (3 inferences/query), variant ${v}"
  fi
  echo "=========================================================="
  echo "[$(date '+%F %T')] verify ${v}  (${desc}) -> ${out}"
  echo "=========================================================="
  python -u run_verification.py \
    --queries-dir "$QUERIES_DIR" \
    --video-dir "$VIDEO_DIR" \
    --output "$out" \
    "${MODE_ARGS[@]}" \
    --verify-rewrite-backend "$BACKEND" \
    --verification-model "$VERIFICATION_MODEL" \
    --qwen-video-reader-backend "$VIDEO_READER" \
    --parallel "$PARALLEL" \
    > "logs/verify_sweep/${v}.log" 2>&1
  echo "[$(date '+%F %T')] done ${v} (exit $?)  log: logs/verify_sweep/${v}.log"
done

echo "All variants done (MODE=${MODE}). Score with:"
echo "  python eval_verification.py --gold ${QUERIES_DIR}/gold.jsonl --results ${OUT_ROOT}/verify_*/verification_results.jsonl --csv ${OUT_ROOT}/verify_metrics.csv"
