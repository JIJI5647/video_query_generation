#!/usr/bin/env bash
# Run the 5 verification-prompt variants over the SAME queries, then you can
# score each against gold with eval_verification.py.
#
# Each prompt -> its own output dir output/verify_<p>/verification_results.jsonl.
#
# Run on the GPU server (qwen3_omni verify) with:
#     export GEMINI_API_KEY=...      # only needed if --verify-rewrite-backend gemini
#     bash run_verification_sweep.sh
# Override defaults via env, e.g.:
#     QUERIES_DIR=data/test5_eval VIDEO_DIR=data/pilot_study PARALLEL=4 bash run_verification_sweep.sh

set -u

QUERIES_DIR="${QUERIES_DIR:-data/test5_eval}"
VIDEO_DIR="${VIDEO_DIR:-data/pilot_study}"
OUT_ROOT="${OUT_ROOT:-output}"
BACKEND="${BACKEND:-qwen3_omni}"          # qwen3_omni | gemini
VIDEO_READER="${VIDEO_READER:-decord}"
PARALLEL="${PARALLEL:-4}"
VERIFICATION_MODEL="${VERIFICATION_MODEL:-gemini-3.1-flash-lite}"

# label -> prompt filename. P1 is the default verification_prompt.txt.
PROMPTS=(
  "p1:verification_prompt.txt"
  "p2_role:verification_prompt_p2_role.txt"
  "p3_fewshot:verification_prompt_p3_fewshot.txt"
  "p4_zscot:verification_prompt_p4_zscot.txt"
  "p5_fewshotcot:verification_prompt_p5_fewshotcot.txt"
)

mkdir -p logs/verify_sweep

for entry in "${PROMPTS[@]}"; do
  label="${entry%%:*}"
  prompt_file="prompts/${entry##*:}"
  out="${OUT_ROOT}/verify_${label}"
  echo "=========================================================="
  echo "[$(date '+%F %T')] verify prompt=${label}  (${prompt_file}) -> ${out}"
  echo "=========================================================="
  python -u run_verification.py \
    --queries-dir "$QUERIES_DIR" \
    --video-dir "$VIDEO_DIR" \
    --output "$out" \
    --using-prompt "$prompt_file" \
    --verify-rewrite-backend "$BACKEND" \
    --verification-model "$VERIFICATION_MODEL" \
    --qwen-video-reader-backend "$VIDEO_READER" \
    --parallel "$PARALLEL" \
    > "logs/verify_sweep/${label}.log" 2>&1
  echo "[$(date '+%F %T')] done ${label} (exit $?)  log: logs/verify_sweep/${label}.log"
done

echo "All prompts done. Score with:"
echo "  python eval_verification.py --gold ${QUERIES_DIR}/gold.jsonl --results ${OUT_ROOT}/verify_*/verification_results.jsonl"
