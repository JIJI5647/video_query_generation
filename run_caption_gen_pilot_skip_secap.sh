#!/usr/bin/env bash
# Resume of run_caption_gen_pilot.sh, skipping secap_qwen entirely.
# qwen_audio_vl and af3_vl already completed (see logs/caption_gen_pilot/*.log).
# secap_qwen has now failed to load twice (2026-07-07 00:52 and 07:37) with no error
# output -- root cause: MotionAudio() in third_party/SECap/model2.py loads 4 separate
# HF checkpoints to CPU (chinese-hubert-large, text2vec, chinese-llama-7b-merged,
# bert-base-chinese) and standalone_inference.py._load_model() THEN torch.load()s a
# redundant 15.8GB model.ckpt state dict on top, all before .to("cuda") -- purely
# CPU/disk-bound, GPU stays at 0% the whole time. On this box that phase alone appears
# to run past the 30-minute GPU-idle reclaim window, and both prior attempts died
# silently (SIGKILL, no traceback) with gpu_keepalive.sh not reliably alive throughout.
# Left for a separate investigation/fix; this run only does the remaining 3 models.
set -u
cd /work/mzha0323/video_query_generation
source env.sh 2>/dev/null

PY=conda_envs/video_env/bin/python
OUT_ROOT=output/caption_gen_pilot
LOG_DIR=logs/caption_gen_pilot
mkdir -p "$LOG_DIR"

run_model() {
  local m="$1"; shift
  echo ""; echo ">>> [$m] START $(date)"
  set +e
  $PY run_caption_generation.py \
    --caption-model "$m" \
    --video-dir data/pilot_study \
    --output "$OUT_ROOT/$m" \
    "$@" \
    > "$LOG_DIR/$m.log" 2>&1
  local rc=$?
  set -e
  [ $rc -eq 0 ] && echo ">>> [$m] DONE ok $(date)" \
                || echo ">>> [$m] FAILED rc=$rc $(date) (see $LOG_DIR/$m.log)"
}

echo "=== pilot caption gen (skip secap_qwen) START $(date) ==="
run_model avocado
run_model timechat
run_model qwen3_omni
echo ""; echo "=== pilot caption gen (skip secap_qwen) END $(date) ==="
