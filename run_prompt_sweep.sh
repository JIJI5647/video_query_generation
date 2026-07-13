#!/usr/bin/env bash
# Prompt-combination sweep: {generation prompt} x {regrounding mode}, evaluated by
# the standard qwen3_omni p7_rolecot verifier. Reuses the Exp3 qwen3_omni captions
# (fixed), so ONLY generation + regrounding + verify differ across cells.
# Swaps prompts/generation_prompt.txt per cell (restores on exit). One cell at a
# time (GPU verify -> no OOM). Resumable: skips a cell whose final_queries exists.
set -u
cd /work/mzha0323/video_query_generation
source env.sh 2>/dev/null

PY=conda_envs/video_env/bin/python
CAPS=output/exp3_unified/captions/qwen3_omni
VIDEO_DIR=data/pilot_study
OUT=output/prompt_sweep
LOG=logs/prompt_sweep
mkdir -p "$OUT" "$LOG"

GEN_PROMPT=prompts/generation_prompt.txt
BACKUP="$LOG/generation_prompt.orig.txt"
cp "$GEN_PROMPT" "$BACKUP"
restore(){ cp "$BACKUP" "$GEN_PROMPT"; echo "[sweep] restored generation_prompt.txt"; }
trap restore EXIT

declare -A GENS=(
  [v12]=prompts/generation_prompt_v12_baseline.txt
  [v13]=prompts/generation_prompt_v13_answerable.txt
  [v14]=prompts/generation_prompt_v14_stateforward.txt
)
REGR=(off full window)

echo "=== prompt sweep START $(date) ==="
for g in v12 v13 v14; do
  for r in "${REGR[@]}"; do
    cell="${g}_${r}"
    dir="$OUT/$cell"
    if [ -f "$dir/final_queries.jsonl" ]; then
      echo ">>> [$cell] SKIP (exists)"; continue
    fi
    echo ">>> [$cell] START $(date)"
    cp "${GENS[$g]}" "$GEN_PROMPT"          # swap in this generation variant
    if [ "$r" = "off" ]; then RG=(--no-regrounding); else RG=(--regrounding-scope "$r"); fi
    set +e
    $PY -u rerun_generation.py \
      --captions-dir "$CAPS" \
      --video-dir "$VIDEO_DIR" \
      --output "$dir" \
      "${RG[@]}" \
      --verify-rewrite-backend qwen_omni_vllm \
      --qwen-vllm-base-url http://localhost:8000/v1 \
      --qwen-vllm-model Qwen/Qwen3-Omni-30B-A3B-Instruct \
      --parallel 4 \
      > "$LOG/$cell.log" 2>&1
    rc=$?
    set -e
    [ $rc -eq 0 ] && echo ">>> [$cell] DONE $(date)" || echo ">>> [$cell] FAILED rc=$rc (see $LOG/$cell.log)"
  done
done
echo "=== prompt sweep END $(date) ==="
