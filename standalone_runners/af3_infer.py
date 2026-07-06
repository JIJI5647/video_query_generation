"""Standalone Audio Flamingo 3 audio captioner — run as a SUBPROCESS.

AF3's HuggingFace integration (``AudioFlamingo3ForConditionalGeneration``) landed
in a newer ``transformers`` than the rest of this pipeline is pinned to, so it
lives in its own venv (``conda_envs/af3_env``) instead of the shared
``video_env``. ``caption_query_test._run_af3_vl`` invokes this script via
``subprocess`` and reads the caption between the ``###CAPTION_START###`` /
``###CAPTION_END###`` markers on stdout.

NON-COMMERCIAL research use only (NVIDIA audio-flamingo-3 license).

Usage: af3_env/bin/python af3_infer.py <audio_path> [model_id]
"""
import sys

_AUDIO_CAPTION_INSTRUCTION = (
    "Listen to this short audio clip. In 1-3 sentences describe HOW the audio "
    "sounds (voice quality, prosody, non-speech sounds such as shouting, crying, "
    "laughing, heavy breathing, silence) — not the exact spoken words. Do NOT "
    "describe anything visual."
)


def main() -> None:
    audio_path = sys.argv[1]
    model_id = sys.argv[2] if len(sys.argv) > 2 else "nvidia/audio-flamingo-3-hf"

    from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)
    model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
        model_id, device_map="auto"
    )

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

    print("###CAPTION_START###")
    print(decoded.strip())
    print("###CAPTION_END###")


if __name__ == "__main__":
    main()
