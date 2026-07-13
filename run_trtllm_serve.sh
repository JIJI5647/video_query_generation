#!/usr/bin/env bash
# Serve Nemotron-3-Nano-Omni on TensorRT-LLM as an OpenAI-compatible endpoint.
#
# This is the intended EFFICIENT-INFERENCE path for the Nemotron p0-p8 verify sweep.
# It requires TensorRT-LLM 1.3.0rc+ (the first release supporting this model's
# Mamba2-hybrid-MoE omni architecture + nano-v3 reasoning parser), which is built
# against CUDA 13 and therefore needs a base NVIDIA driver >= R580 (CUDA 13 forward
# compatibility requires R580; R570 is the floor for cuda-compat-13). On a host with
# an older driver (e.g. R550) trtllm-serve fails at CUDA init -- use vLLM instead
# (same OpenAI API, so run_nemotron_sweep.sh is unchanged; see docs/progress_log.md).
#
#   bash run_trtllm_serve.sh              # foreground
#   MODEL=...-FP8 PORT=8000 bash run_trtllm_serve.sh
set -u

VENV="${VENV:-/work/mzha0323/trtllm_venv}"
MODEL="${MODEL:-nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
CFG="${CFG:-/tmp/nano_v3.yaml}"

cat > "$CFG" <<EOF
kv_cache_config:
  enable_block_reuse: false
  free_gpu_memory_fraction: 0.80
  mamba_ssm_cache_dtype: float32
max_batch_size: 128
EOF

echo "[trtllm-serve] model=$MODEL host=$HOST port=$PORT venv=$VENV"
PYTORCH_ALLOC_CONF=expandable_segments:True \
HF_HOME="${HF_HOME:-/work/mzha0323/hf_cache}" \
"$VENV/bin/trtllm-serve" serve "$MODEL" \
  --host "$HOST" --port "$PORT" --trust_remote_code \
  --reasoning_parser nano-v3 --tool_parser qwen3_coder \
  --extra_llm_api_options "$CFG"
