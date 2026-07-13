#!/usr/bin/env bash
# Regenerate NEW unified visual+audio captions for ALL 6 caption models over the
# full 19-video pilot set. vLLM-accelerated Qwen3-VL video half for the three
# two-branch models (qwen_audio_vl / af3_vl / secap_qwen) via --vllm-video
# (Qwen3-VL runs under vLLM in conda_envs/vllm_env). Single-model captioners
# (avocado / timechat / qwen3_omni) have no vLLM caption path -> transformers.
# One model at a time (GPU freed between models). Per-segment cache -> resumable.
set -u
cd /work/mzha0323/video_query_generation
source env.sh 2>/dev/null
# Force decord video decode (fast, C++). torchcodec fails to load libtorchcodec ->
# it otherwise falls back to torchvision (pure-CPU, ~2-3 min/segment, GPU idle).
export FORCE_QWENVL_VIDEO_READER=decord
PY=conda_envs/video_env/bin/python
OUT=output/caption_gen_unified19
LOG=logs/caption_gen_unified19
mkdir -p "$OUT" "$LOG"

# --vllm-video dropped: the bottleneck was CPU video decode (torchvision), not the
# video model — decord fixes that. vLLM added a slow startup + HTTP overhead
# without addressing the real bottleneck. Empty = all models use in-process transformers.
VLLM_VIDEO_MODELS=" "
MODELS=(qwen_audio_vl af3_vl avocado timechat qwen3_omni)   # secap dropped; qwen3_omni(30B) last. nemotron_omni handled separately (served).

echo "=== unified 19v caption gen (6 models, vLLM video) START $(date) ==="
for m in "${MODELS[@]}"; do
  f="$OUT/$m/raw_captions.jsonl"
  if [ -f "$f" ] && [ "$(python3 -c "import json;print(len(set(json.loads(l)['video_id'] for l in open('$f') if l.strip())))" 2>/dev/null)" = "19" ]; then
    echo ">>> [$m] SKIP (19 videos done)"; continue
  fi
  EXTRA=()
  case "$VLLM_VIDEO_MODELS" in *" $m "*) EXTRA=(--vllm-video); echo ">>> [$m] using --vllm-video";; esac
  echo ">>> [$m] START $(date)"
  set +e
  $PY -u run_caption_generation.py --caption-model "$m" \
    --video-dir data/pilot_study --output "$OUT/$m" "${EXTRA[@]}" \
    > "$LOG/$m.log" 2>&1
  rc=$?; set -e
  [ $rc -eq 0 ] && echo ">>> [$m] DONE $(date)" || echo ">>> [$m] FAILED rc=$rc (see $LOG/$m.log)"
done
echo "=== unified 19v caption gen END $(date) ==="
