#!/usr/bin/env bash
# p0..p8 per-dimension verifier ablation on Nemotron-3-Nano-Omni, mirroring
# run_verification_sweep.sh but against the Nemotron OpenAI-compatible server
# (trtllm-serve OR vllm serve -- identical API, so this script is engine-agnostic).
#
# The server must already be running (run_trtllm_serve.sh, or a vllm serve). Each
# variant -> its own dir ${OUT_ROOT}/verify_<variant>/. Score afterwards with
# eval_verification.py against gold, exactly like the Qwen Thinking sweep.
#
#   bash run_nemotron_sweep.sh
#   OUT_ROOT=output/nemotron_sweep BASE_URL=http://0.0.0.0:8000/v1 PARALLEL=4 bash run_nemotron_sweep.sh
set -u

QUERIES_DIR="${QUERIES_DIR:-data/test5_eval}"
VIDEO_DIR="${VIDEO_DIR:-data/pilot_study}"
OUT_ROOT="${OUT_ROOT:-output/nemotron_sweep}"
BASE_URL="${BASE_URL:-http://0.0.0.0:8000/v1}"
MODEL="${MODEL:-nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
PARALLEL="${PARALLEL:-4}"
PYTHON="${PYTHON:-$(pwd)/conda_envs/video_env/bin/python}"

VARIANTS=(
  p0_norule p1_rule p2_role p3_fewshot p4_zscot
  p5_fewshotcot p6_rolefewshot p7_rolecot p8_rawcot
)

mkdir -p logs/nemotron_sweep
for v in "${VARIANTS[@]}"; do
  out="${OUT_ROOT}/verify_${v}"
  echo "=========================================================="
  echo "[$(date '+%F %T')] verify ${v} (per-dimension) -> ${out}"
  echo "=========================================================="
  "$PYTHON" -u run_verification.py \
    --queries-dir "$QUERIES_DIR" \
    --video-dir "$VIDEO_DIR" \
    --output "$out" \
    --per-dimension --variant "$v" \
    --verify-rewrite-backend nemotron \
    --nemotron-base-url "$BASE_URL" \
    --nemotron-model "$MODEL" \
    --nemotron-max-tokens "$MAX_TOKENS" \
    --parallel "$PARALLEL" \
    > "logs/nemotron_sweep/${v}.log" 2>&1
  echo "[$(date '+%F %T')] done ${v} (exit $?)  log: logs/nemotron_sweep/${v}.log"
done

echo "All variants done. Score with:"
echo "  python eval_verification.py --gold ${QUERIES_DIR}/gold.jsonl \\"
echo "    --results ${OUT_ROOT}/verify_*/verification_results.jsonl \\"
echo "    --csv ${OUT_ROOT}/verify_metrics_nemotron.csv"
