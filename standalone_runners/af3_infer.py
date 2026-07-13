"""Standalone Audio Flamingo 3 audio captioner — run as a SUBPROCESS.

AF3's HuggingFace integration (``AudioFlamingo3ForConditionalGeneration``) landed
in a newer ``transformers`` than the rest of this pipeline is pinned to, so it
lives in its own venv (``conda_envs/af3_env``) instead of the shared
``video_env``. The caller invokes this script via ``subprocess`` and reads each
caption between the ``###CAPTION_START###`` / ``###CAPTION_END###`` markers on
stdout.

NON-COMMERCIAL research use only (NVIDIA audio-flamingo-3 license).

Two modes:
  - single-file:  af3_env/bin/python af3_infer.py <audio_path> [model_id]
      loads the model, captions one file, exits. (used by the per-segment test tool)
  - server:       af3_env/bin/python af3_infer.py --server [model_id]
      loads the model ONCE, then loops reading one audio path per line from stdin,
      emitting one caption block per line, until stdin closes. Lets the batch
      captioner caption a whole run without reloading the ~model per segment.
"""
import sys

_AUDIO_CAPTION_INSTRUCTION = (
    "Listen to this short audio clip. In AT MOST 2 sentences describe HOW the "
    "audio sounds (voice quality, prosody, non-speech sounds such as shouting, "
    "crying, laughing, heavy breathing, silence) — not the exact spoken words. "
    "Do NOT name or infer any emotion: describe only the acoustic qualities "
    "themselves (pitch, loudness, pace, breathiness, gasp, tremor, sigh), NEVER "
    "an emotion the voice 'expresses' or 'conveys' (do not write 'surprised', "
    "'angry', 'sad', 'shocked', etc.). Emotion is decided in a later, separate "
    "stage. Do NOT describe anything visual. Hard limit: 2 sentences."
)


def _caption(processor, model, audio_path: str) -> str:
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": _AUDIO_CAPTION_INSTRUCTION},
                {"type": "audio", "path": audio_path},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        conversation, tokenize=True, add_generation_prompt=True, return_dict=True,
    ).to(model.device, dtype=model.dtype)
    outputs = model.generate(**inputs, max_new_tokens=500)
    decoded = processor.decode(
        outputs[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    return decoded.strip()


def _emit(text: str) -> None:
    print("###CAPTION_START###")
    print(text)
    print("###CAPTION_END###")
    sys.stdout.flush()


def main() -> None:
    server_mode = len(sys.argv) > 1 and sys.argv[1] == "--server"
    if server_mode:
        model_id = sys.argv[2] if len(sys.argv) > 2 else "nvidia/audio-flamingo-3-hf"
    else:
        audio_path = sys.argv[1]
        model_id = sys.argv[2] if len(sys.argv) > 2 else "nvidia/audio-flamingo-3-hf"

    from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)
    model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
        model_id, device_map="auto"
    )

    if not server_mode:
        _emit(_caption(processor, model, audio_path))
        return

    # Server mode: signal readiness, then caption one path per stdin line.
    print("###READY###")
    sys.stdout.flush()
    for line in sys.stdin:
        path = line.strip()
        if not path:
            continue
        try:
            _emit(_caption(processor, model, path))
        except Exception as e:  # keep the server alive; report per-item failure
            print("###CAPTION_START###")
            print(f"__ERROR__: {e}")
            print("###CAPTION_END###")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
