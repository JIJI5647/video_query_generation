"""Persistent Qwen3-VL video-caption server backed by vLLM (offline engine).

Runs in ``conda_envs/vllm_env`` (vllm 0.14 + torch cu126 — a CUDA-12 stack that the
GPU driver, capped at CUDA 12.6, can run; see docs/vllm_captioning.md). Loads Qwen3-VL
ONCE, then serves one caption per stdin line so the shared pipeline (in ``video_env``)
can offload the video half of qwen_audio_vl / af3_vl / secap_qwen to it without pulling
vLLM into the shared env.

Protocol (mirrors standalone_runners/af3_infer.py --server / SubprocessAudioSession):
  * prints ``###READY###`` once the engine is loaded,
  * then for each line of stdin (one absolute video-clip path) prints the caption
    between ``###CAPTION_START###`` / ``###CAPTION_END###`` markers,
  * on a per-item failure prints ``__ERROR__: <msg>`` inside those markers (the caller
    raises), so one bad clip never kills the resident server.

Single-shot mode (no ``--server``): caption one clip path given as argv and exit — handy
for debugging. The visual-caption instruction is kept byte-identical to
emotion_query_pipeline.caption_query_test.VIDEO_CAPTION_INSTRUCTION so vLLM output is a
drop-in replacement for the transformers path.
"""
import argparse
import os
import sys

# Keep vLLM quiet on stderr. The parent drains our stderr in a thread so a full pipe
# can't deadlock us, but less noise is faster and keeps error tails readable.
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
os.environ.setdefault("TQDM_DISABLE", "1")

# Byte-identical to caption_query_test.VIDEO_CAPTION_INSTRUCTION (video-only, no emotion).
VIDEO_CAPTION_INSTRUCTION = (
    "Watch this short video clip (no audio). In 2-4 sentences describe ONLY what "
    "is visible: the people (appearance, position, visibility), their observable "
    "actions, the scene, and any observable facial cues, body cues, posture, "
    "gestures and gaze. Do NOT name or infer any emotion, and do NOT describe "
    "sound or speech. Describe only what is visible."
)

READY = "###READY###"
CAP_START = "###CAPTION_START###"
CAP_END = "###CAPTION_END###"


class Qwen3VLVllmCaptioner:
    def __init__(self, model_path: str, max_new_tokens: int = 1024,
                 gpu_memory_utilization: float = 0.85, max_model_len: int = 32768) -> None:
        from vllm import LLM, SamplingParams
        from transformers import AutoProcessor

        self._llm = LLM(
            model=model_path,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            limit_mm_per_prompt={"video": 1, "image": 0},
        )
        self._processor = AutoProcessor.from_pretrained(model_path)
        self._sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)

    def _build(self, video_path: str):
        from vllm.assets.video import video_to_ndarrays, video_get_metadata

        frames = video_to_ndarrays(video_path, -1)
        meta = video_get_metadata(video_path, -1)
        messages = [{
            "role": "user",
            "content": [
                {"type": "video", "video": video_path},
                {"type": "text", "text": VIDEO_CAPTION_INSTRUCTION},
            ],
        }]
        prompt = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return {"prompt": prompt, "multi_modal_data": {"video": (frames, meta)}}

    def caption(self, video_path: str) -> str:
        out = self._llm.generate([self._build(video_path)], self._sp, use_tqdm=False)
        return (out[0].outputs[0].text if out and out[0].outputs else "").strip()


def _emit(text: str) -> None:
    print(CAP_START)
    print(text)
    print(CAP_END)
    sys.stdout.flush()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video_path", nargs="?", default=None)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--server", action="store_true",
                    help="Persistent mode: load once, caption one stdin path per line.")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=32768)
    args = ap.parse_args()

    captioner = Qwen3VLVllmCaptioner(
        args.model, max_new_tokens=args.max_new_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )

    if not args.server:
        if not args.video_path:
            print("usage: qwen3vl_vllm_server.py <video.mp4> | --server", file=sys.stderr)
            sys.exit(2)
        _emit(captioner.caption(args.video_path))
        return

    print(READY)
    sys.stdout.flush()
    for line in sys.stdin:
        path = line.strip()
        if not path:
            continue
        try:
            _emit(captioner.caption(path))
        except Exception as exc:  # never let one clip kill the resident server
            _emit(f"__ERROR__: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
