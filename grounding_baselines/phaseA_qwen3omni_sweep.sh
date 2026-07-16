#!/usr/bin/env bash
# Phase A of the 313-pool prompt sweep: Qwen3-Omni x {p0,p1_strict,p2_cot,
# p3_emotion,p4_grid} on the 213-query dev split (lowres videos, serve must
# already be up). Resumable: run_nemotron.py skips cached query_ids.
set -u
cd /work/mzha0323/video_query_generation
source env.sh 2>/dev/null
PY=conda_envs/video_env/bin/python
DEV=output/grounding_eval/pool_dev_lowres.jsonl
for P in p0 p1_strict p2_cot p3_emotion p4_grid; do
  OUT=output/grounding_eval/sweep_qwen3omni_$P
  echo ">>> [$P] START $(date)"
  $PY grounding_baselines/run_nemotron.py \
    --model "Qwen/Qwen3-Omni-30B-A3B-Instruct" --thinking none \
    --gold "$DEV" --output "$OUT" --prompt "$P" --parallel 4 \
    > "logs/grounding_eval/sweep_qwen3omni_$P.log" 2>&1 || echo "[$P] FAILED"
  echo ">>> [$P] DONE $(date)"
done
echo "PHASE A COMPLETE $(date)"
