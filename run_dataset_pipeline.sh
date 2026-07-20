#!/usr/bin/env bash
# =====================================================================
# Full emotion-query DATASET-GENERATION pipeline for one video folder.
# caption (vLLM Qwen3-Omni) -> emotion events (Gemini, no-cap) -> query
# gen (Gemini) -> refine (Qwen3-Omni) -> guardrail -> verify (Qwen3-Omni).
# Resumable: every stage is skipped if its output already exists.
#
# Usage:
#   bash run_dataset_pipeline.sh <video-dir> <seg-seconds> <output-dir> [sample-N]
# Example (50 videos from the dev split, 5s segments):
#   bash run_dataset_pipeline.sh data/gdrive_dataset/video_splits/dev 5 output/gen_dev50 50
#
# Requirements: env.sh with GEMINI_API_KEY; conda_envs/video_env; the vLLM venv
# used by run_vllm_serve_qwen.sh; ffmpeg on PATH. Runs on the single H200.
# =====================================================================
set -u
cd "$(dirname "$0")"
source env.sh 2>/dev/null
PY=conda_envs/video_env/bin/python
QOMNI="Qwen/Qwen3-Omni-30B-A3B-Instruct"

VID_DIR="${1:?usage: run_dataset_pipeline.sh <video-dir> <seg-seconds> <output-dir> [sample-N]}"
SEG="${2:?need seg-seconds (e.g. 5)}"
OUT="${3:?need output-dir}"
SAMPLE="${4:-0}"                       # 0 = all videos in the folder
mkdir -p "$OUT" logs
LOG="logs/$(basename "$OUT")"; mkdir -p "$LOG"

echo "=== dataset pipeline START $(date) | vids=$VID_DIR seg=${SEG}s out=$OUT sample=$SAMPLE ==="

# --- pick video ids -------------------------------------------------
if [ ! -f "$OUT/videos.txt" ]; then
  ls "$VID_DIR"/*.mp4 2>/dev/null | xargs -n1 basename | sed 's/\.mp4$//' > "$OUT/videos.txt"
  if [ "$SAMPLE" -gt 0 ]; then
    # deterministic sample: first N by sorted name (change if you want random)
    sort "$OUT/videos.txt" | head -n "$SAMPLE" > "$OUT/videos.sel"
  else
    cp "$OUT/videos.txt" "$OUT/videos.sel"
  fi
fi
VIDS=$(paste -sd, "$OUT/videos.sel")
NV=$(wc -l < "$OUT/videos.sel")
echo ">>> $NV videos selected"

# --- serve helpers --------------------------------------------------
serve_qomni() {
  curl -s -m 3 http://localhost:8000/v1/models 2>/dev/null | grep -q Instruct && return 0
  MODEL="$QOMNI" PORT=8000 GPU_UTIL=0.7 MAX_LEN=65536 \
    nohup bash run_vllm_serve_qwen.sh > "$LOG/serve.log" 2>&1 &
  for i in $(seq 1 120); do
    curl -s -m 3 http://localhost:8000/v1/models 2>/dev/null | grep -q Instruct && return 0
    sleep 10
  done
  return 1
}
stop_serve() {
  APIPID=$(ps -eo pid,cmd | grep "[v]llm serve" | awk '{print $1}' | head -1)
  [ -n "${APIPID:-}" ] && kill -TERM "$APIPID"
  pkill -TERM -f "VLLM::[E]ngineCore" 2>/dev/null; sleep 15
}
trap 'stop_serve' EXIT

echo ">>> [serve] Qwen3-Omni $(date)"; serve_qomni || { echo "SERVE FAILED"; exit 1; }

# --- 1. captions (vLLM Qwen3-Omni via the served nemotron route) ----
if [ ! -f "$OUT/captions/raw_captions.jsonl" ]; then
  echo ">>> [1/5] caption @${SEG}s $(date)"
  $PY -u run_caption_generation.py --caption-model nemotron_omni --video-dir "$VID_DIR" \
    --video-ids "$VIDS" --segment-seconds "$SEG" --stride "$SEG" --output "$OUT/captions" \
    --nemotron-model "$QOMNI" --nemotron-base-url http://localhost:8000/v1 \
    --nemotron-no-thinking --caption-parallel 8 > "$LOG/caption.log" 2>&1 \
    || { echo "CAPTION FAILED (see $LOG/caption.log)"; exit 1; }
  # PATCH: run_caption_generation omits video_id in segments.jsonl; derive from clip_path
  $PY - "$OUT/captions/segments.jsonl" <<'PYEOF'
import json, sys
from pathlib import Path
fn=sys.argv[1]; rows=[json.loads(l) for l in open(fn)]; ch=0
for r in rows:
    if not r.get("video_id"):
        p=Path(r["clip_path"]).parts; r["video_id"]=p[p.index("processed_segments")+1]; ch+=1
open(fn,"w").write("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows))
print(f"patched video_id in segments: {ch}/{len(rows)}")
PYEOF
else echo ">>> [1/5] caption SKIP"; fi

# --- 2. emotion events (Gemini, cap removed) ------------------------
if [ ! -f "$OUT/events/emotion_events.jsonl" ]; then
  echo ">>> [2/5] emotion events (Gemini) $(date)"
  $PY mm_event_pilot.py --backend gemini-text --videos "$VIDS" \
    --captions-dir "$OUT/captions" --output "$OUT/events" > "$LOG/events.log" 2>&1 \
    || { echo "EVENTS FAILED"; exit 1; }
else echo ">>> [2/5] events SKIP"; fi

# --- 3. query generation (Gemini text + disambiguation) -------------
if [ ! -f "$OUT/gemini_base/initial_queries.jsonl" ]; then
  echo ">>> [3/5] query generation (Gemini) $(date)"
  $PY gen_text_from_events.py --events-dir "$OUT/events" --videos "$VIDS" \
    --captions-dir "$OUT/captions" --output "$OUT/gemini_base" > "$LOG/gen.log" 2>&1 \
    || { echo "GEN FAILED"; exit 1; }
else echo ">>> [3/5] gen SKIP"; fi

# --- 4. refine (Qwen3-Omni watches clips) + guardrail --------------
if [ ! -f "$OUT/final/initial_queries.jsonl" ]; then
  echo ">>> [4/5] refine + guardrail (Qwen3-Omni) $(date)"
  $PY hybrid_refine.py --base "$OUT/gemini_base" --model "$QOMNI" --thinking none \
    --output "$OUT/refine" > "$LOG/refine.log" 2>&1 || { echo "REFINE FAILED"; exit 1; }
  $PY hybrid_guardrail.py --in "$OUT/refine" --out "$OUT/final" > "$LOG/guardrail.log" 2>&1 \
    || { echo "GUARDRAIL FAILED"; exit 1; }
else echo ">>> [4/5] refine+guardrail SKIP"; fi

# --- 5. verify (Qwen3-Omni, p7_rolecot per-dimension) --------------
if [ ! -f "$OUT/final/verification_results.jsonl" ]; then
  echo ">>> [5/5] verify (Qwen3-Omni) $(date)"
  $PY run_verification.py --queries-dir "$OUT/final" --video-dir "$VID_DIR" --output "$OUT/final" \
    --verify-rewrite-backend qwen_omni_vllm --qwen-vllm-base-url http://localhost:8000/v1 \
    --qwen-vllm-model "$QOMNI" --per-dimension --variant p7_rolecot --parallel 4 \
    > "$LOG/verify.log" 2>&1 || { echo "VERIFY FAILED"; exit 1; }
else echo ">>> [5/5] verify SKIP"; fi

stop_serve; trap - EXIT

# --- summary --------------------------------------------------------
$PY - "$OUT" <<'PYEOF'
import json, sys
from pathlib import Path
OUT=Path(sys.argv[1])
q=[json.loads(l) for l in open(OUT/"final/initial_queries.jsonl")]
nq=sum(len(r["queries"]) for r in q)
ver=[json.loads(l) for l in open(OUT/"final/verification_results.jsonl")]
npass=sum(1 for x in ver if x["decision"]=="pass")
print(f"\n===== DONE: {nq} queries, {npass} pass ({npass/len(ver):.0%}) over {len(q)} videos =====")
print(f"outputs in {OUT}/final/{{initial_queries.jsonl, verification_results.jsonl}}")
PYEOF
echo "=== dataset pipeline END $(date) ==="
