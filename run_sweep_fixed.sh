#!/usr/bin/env bash
# CONTROLLED prompt sweep: query set is FIXED per generation prompt; only the
# re-grounding scope varies within it. For each gen in {v12,v13,v14} the base
# query set is that gen's *_off initial_queries.jsonl (pure generation). Each
# scope (off/full/window) re-grounds THAT SAME set, then verifies via the
# qwen_omni_vllm endpoint (p7_rolecot per-dimension). So off/full/window are a
# clean isolation of the regrounding effect (identical query text + generation
# grounding); v12/v13/v14 differ only because their generation prompts differ.
# Resumable: skips a cell whose verification_results.jsonl exists.
set -u
cd /work/mzha0323/video_query_generation
source env.sh 2>/dev/null

PY=conda_envs/video_env/bin/python
CAPS=output/exp3_unified/captions/qwen3_omni
VIDEO_DIR=data/pilot_study
OUT=output/sweep_fixed
LOG=logs/sweep_fixed
mkdir -p "$OUT" "$LOG"

GENS=(v12 v13 v14)
SCOPES=(off full window)

echo "=== controlled sweep START $(date) ==="
for g in "${GENS[@]}"; do
  BASE="output/prompt_sweep/${g}_off"
  if [ ! -f "$BASE/initial_queries.jsonl" ]; then
    echo ">>> [$g] MISSING base $BASE — skip"; continue
  fi
  for s in "${SCOPES[@]}"; do
    cell="${g}_${s}"; dir="$OUT/$cell"
    if [ -f "$dir/verification_results.jsonl" ]; then
      echo ">>> [$cell] SKIP (exists)"; continue
    fi
    echo ">>> [$cell] START $(date)"
    mkdir -p "$dir"
    # 1. Re-ground the FIXED base query set at this scope.
    $PY -u apply_regrounding.py \
      --base-queries "$BASE" --captions-dir "$CAPS" \
      --scope "$s" --output "$dir" > "$LOG/${cell}.reground.log" 2>&1
    if [ $? -ne 0 ]; then echo ">>> [$cell] REGROUND FAILED (see $LOG/${cell}.reground.log)"; continue; fi
    # 2. Verify (same fixed queries, only grounding changed).
    set +e
    $PY -u run_verification.py \
      --queries-dir "$dir" --video-dir "$VIDEO_DIR" --output "$dir" \
      --verify-rewrite-backend qwen_omni_vllm \
      --qwen-vllm-base-url http://localhost:8000/v1 \
      --qwen-vllm-model Qwen/Qwen3-Omni-30B-A3B-Instruct \
      --per-dimension --variant p7_rolecot --parallel 4 \
      > "$LOG/${cell}.verify.log" 2>&1
    rc=$?; set -e
    [ $rc -eq 0 ] && echo ">>> [$cell] DONE $(date)" || echo ">>> [$cell] VERIFY FAILED rc=$rc (see $LOG/${cell}.verify.log)"
  done
done
echo "=== controlled sweep END $(date) ==="
