#!/usr/bin/env bash
# Serve Nemotron-3-Nano-Omni on vLLM as an OpenAI-compatible endpoint.
#
# vLLM is the runnable EFFICIENT-INFERENCE fallback on this host: TensorRT-LLM 1.3.0rc
# (the version that supports this model) needs CUDA 13 / driver >= R580, which this box
# (driver 550.163) can't provide. vLLM 0.20.0 also supports the model AND ships a
# **CUDA 12.9** build (vllm-0.20.0+cu129), which runs on the R550 driver via CUDA 12.x
# minor-version compatibility (driver >= 525). Same OpenAI API as trtllm-serve, so
# run_nemotron_sweep.sh is unchanged. Installed into /work/mzha0323/vllm020_venv.
#
#   bash run_vllm_serve.sh                     # foreground
#   MODEL=...-FP8 PORT=8000 bash run_vllm_serve.sh
set -u

VENV="${VENV:-/work/mzha0323/vllm020_venv}"
MODEL="${MODEL:-nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8}"
# NB: use SERVE_HOST, not HOST — conda pre-sets HOST=x86_64-conda-linux-gnu (build
# triple), which would otherwise leak in as the bind address and crash on gaierror.
SERVE_HOST="${SERVE_HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MAX_LEN="${MAX_LEN:-32768}"
MAX_SEQS="${MAX_SEQS:-16}"
GPU_UTIL="${GPU_UTIL:-0.85}"
export HF_HOME="${HF_HOME:-/work/mzha0323/hf_cache}"

# vLLM's EngineCore subprocess JIT-compiles the Mamba/causal-conv1d kernels (this is a
# Mamba2-hybrid MoE model) and shells out to `ninja` via PATH lookup. We invoke vllm by
# absolute path without activating the venv, so venv/bin (which has ninja) isn't on PATH
# -> "FileNotFoundError: 'ninja'" during KV-cache profiling. Put venv/bin on PATH.
export PATH="$VENV/bin:$PATH"

# flashinfer JIT-compiles the FP8 CUTLASS MoE kernel (fused_moe_90) with the conda nvcc,
# but that nvcc's include path lacks the CUDA library headers (cublasLt.h etc.) -> the
# compile dies with "fatal error: cublasLt.h: No such file or directory". The headers +
# .so's DO exist inside the pip nvidia-* packages; wire every nvidia/<lib>/{include,lib}
# onto CPATH / LIBRARY_PATH / LD_LIBRARY_PATH so nvcc's preprocessor and linker find them.
NV_SITE="$VENV/lib/python3.12/site-packages/nvidia"
for d in "$NV_SITE"/*/include; do [ -d "$d" ] && CPATH="$d:${CPATH:-}"; done
for d in "$NV_SITE"/*/lib; do [ -d "$d" ] && LIBRARY_PATH="$d:${LIBRARY_PATH:-}" && LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH:-}"; done
export CPATH LIBRARY_PATH LD_LIBRARY_PATH

echo "[vllm serve] model=$MODEL host=$SERVE_HOST port=$PORT venv=$VENV HF_HOME=$HF_HOME"
"$VENV/bin/vllm" serve "$MODEL" \
  --host "$SERVE_HOST" --port "$PORT" \
  --max-model-len "$MAX_LEN" \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --gpu-memory-utilization "$GPU_UTIL" \
  --video-pruning-rate 0.5 \
  --max-num-seqs "$MAX_SEQS" \
  --allowed-local-media-path / \
  --media-io-kwargs '{"video": {"fps": 2, "num_frames": 256}}' \
  --reasoning-parser nemotron_v3 \
  --tool-call-parser qwen3_coder \
  --enable-auto-tool-choice \
  --kv-cache-dtype fp8
