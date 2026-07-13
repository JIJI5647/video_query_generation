#!/usr/bin/env bash
# Serve Qwen3-Omni on vLLM as an OpenAI-compatible endpoint, for the
# `qwen_omni_vllm` verify/rewrite backend (run_verification.py / rerun_generation.py).
#
# Sibling of run_vllm_serve.sh (which serves Nemotron-3-Nano-Omni) — same vLLM
# 0.20.0 venv and the SAME flashinfer/CPATH/ninja workarounds are needed here
# because they fix the MoE JIT compile on this box, not anything Nemotron-specific
# (Qwen3-Omni-30B-A3B is also a sparse-MoE model). Confirmed to load and generate
# under vLLM 0.20.0 with gpu_memory_utilization~=0.7, max_model_len 8192,
# enforce_eager=True.
#
# Differs from run_vllm_serve.sh in the serve flags only: no --reasoning-parser
# nemotron_v3 / --tool-call-parser qwen3_coder (Nemotron-specific), and the
# model/mem/context-length knobs used in the confirmed-working config above.
#
#   bash run_vllm_serve_qwen.sh                     # foreground
#   MODEL=...-Thinking PORT=8000 bash run_vllm_serve_qwen.sh
set -u

VENV="${VENV:-/work/mzha0323/vllm020_venv}"
MODEL="${MODEL:-Qwen/Qwen3-Omni-30B-A3B-Instruct}"
# NB: use SERVE_HOST, not HOST — conda pre-sets HOST=x86_64-conda-linux-gnu (build
# triple), which would otherwise leak in as the bind address and crash on gaierror.
SERVE_HOST="${SERVE_HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MAX_LEN="${MAX_LEN:-16384}"
MAX_SEQS="${MAX_SEQS:-8}"
GPU_UTIL="${GPU_UTIL:-0.7}"
export HF_HOME="${HF_HOME:-/work/mzha0323/hf_cache}"

# vLLM's EngineCore subprocess JIT-compiles kernels for this MoE model and shells
# out to `ninja` via PATH lookup. We invoke vllm by absolute path without
# activating the venv, so venv/bin (which has ninja) isn't on PATH ->
# "FileNotFoundError: 'ninja'" during KV-cache profiling. Put venv/bin on PATH.
export PATH="$VENV/bin:$PATH"

# flashinfer JIT-compiles the FP8 CUTLASS MoE kernel (fused_moe_90) with the conda
# nvcc, but that nvcc's include path lacks the CUDA library headers (cublasLt.h
# etc.) -> the compile dies with "fatal error: cublasLt.h: No such file or
# directory". The headers + .so's DO exist inside the pip nvidia-* packages; wire
# every nvidia/<lib>/{include,lib} onto CPATH / LIBRARY_PATH / LD_LIBRARY_PATH so
# nvcc's preprocessor and linker find them.
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
  --enforce-eager \
  --max-num-seqs "$MAX_SEQS" \
  --allowed-local-media-path / \
  --media-io-kwargs '{"video": {"fps": 2, "num_frames": 256}}'
