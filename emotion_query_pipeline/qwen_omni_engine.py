"""Qwen3-Omni transformers engine + verify/rewrite client.

The 30B Qwen3-Omni model is used in this pipeline ONLY for the verify/rewrite
stage (it watches a query's grounded segment clip(s) locally — no Files API
upload). Captioning is handled by the dedicated observation backends.

LAZY EVERYTHING: ``torch`` / ``transformers`` / ``qwen_omni_utils`` are imported,
and the model loaded, only inside ``_ensure_model`` (first inference). Importing
this module and constructing the engine touch no GPU and download no weights.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .caption_utils import CaptionParseError, extract_caption_json

DEFAULT_MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

# Sampling defaults (mapped to the transformers ``thinker_*`` generate kwargs).
DEFAULT_SAMPLING_PARAMS: Dict[str, Any] = {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "max_tokens": 2048,
}


@dataclass
class Qwen3OmniCaptioner:
    """Qwen3-Omni transformers engine. Loads on first use; import-safe to construct.

    Despite the historical name, this is now used only as a generic batched
    generate engine for the verify/rewrite ``QwenOmniLLMClient``.
    """

    model_path: str = DEFAULT_MODEL_PATH
    sampling_params: Dict[str, Any] = field(
        default_factory=lambda: dict(DEFAULT_SAMPLING_PARAMS)
    )
    use_audio_in_video: bool = True
    # Force the qwen_omni_utils video reader so it never tries torchcodec
    # (which often fails on mismatched CUDA/ffmpeg). Set via FORCE_QWENVL_VIDEO_READER.
    video_reader_backend: str = "torchvision"
    device_map: str = "auto"
    attn_implementation: Optional[str] = None  # e.g. "flash_attention_2"

    _model: Any = field(default=None, init=False, repr=False)
    _processor: Any = field(default=None, init=False, repr=False)

    # -- model loading -------------------------------------------------------
    def _ensure_model(self) -> None:
        """Load the model + processor exactly once. Heavy imports live here."""
        if self._model is not None:
            return
        if self.video_reader_backend:
            os.environ["FORCE_QWENVL_VIDEO_READER"] = self.video_reader_backend
        import time as _time
        t0 = _time.perf_counter()
        print(f"[model] loading {self.model_path} (transformers)...")
        self._ensure_model_transformers()
        print(f"[model] loaded in {_time.perf_counter() - t0:.1f}s "
              f"(one-time; reused for all calls)")

    def _ensure_model_transformers(self) -> None:
        from transformers import (
            Qwen3OmniMoeForConditionalGeneration,
            Qwen3OmniMoeProcessor,
        )

        load_kwargs: Dict[str, Any] = {
            "dtype": "auto",
            "device_map": self.device_map,
            "trust_remote_code": True,
        }
        if self.attn_implementation:
            load_kwargs["attn_implementation"] = self.attn_implementation
        self._model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            self.model_path, **load_kwargs
        )
        # Text-only output — drop the audio-generating talker to save memory/time.
        if hasattr(self._model, "disable_talker"):
            try:
                self._model.disable_talker()
            except Exception:
                pass
        self._processor = Qwen3OmniMoeProcessor.from_pretrained(self.model_path)

    @staticmethod
    def _build_messages_multi(prompt_text: str, clip_paths: List[str]) -> list:
        """One user turn: N segment videos (audio included) then the text prompt.

        Empty ``clip_paths`` is a text-only turn.
        """
        content = [{"type": "video", "video": p} for p in clip_paths if p]
        content.append({"type": "text", "text": prompt_text})
        return [{"role": "user", "content": content}]

    # -- inference -----------------------------------------------------------
    def generate(self, messages: list) -> str:
        out = self.generate_many([messages])
        return out[0] if out else ""

    def generate_many(self, messages_list: List[list]) -> List[str]:
        """Run inference on N pre-built conversations in ONE batched call.

        Output order matches input order; each conversation is independent.
        """
        if not messages_list:
            return []
        self._ensure_model()
        return self._generate_transformers_batch(messages_list)

    # -- transformers helpers ------------------------------------------------
    def _prep_transformers_inputs(self, conversations):
        from qwen_omni_utils import process_mm_info  # lazy

        text = self._processor.apply_chat_template(
            conversations, add_generation_prompt=True, tokenize=False
        )
        audios, images, videos = process_mm_info(
            conversations, use_audio_in_video=self.use_audio_in_video
        )
        inputs = self._processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=self.use_audio_in_video,
        )
        return inputs.to(self._model.device).to(self._model.dtype)

    def _run_transformers_generate(self, inputs) -> List[str]:
        import torch

        sp = self.sampling_params
        base_kwargs: Dict[str, Any] = {
            "return_audio": False,
            "thinker_return_dict_in_generate": True,
            "use_audio_in_video": self.use_audio_in_video,
        }
        sampling_kwargs: Dict[str, Any] = {
            "thinker_max_new_tokens": sp.get("max_tokens", 2048),
            "thinker_do_sample": sp.get("temperature", 0.0) > 0,
            "thinker_temperature": sp.get("temperature", 0.6),
            "thinker_top_p": sp.get("top_p", 0.95),
            "thinker_top_k": sp.get("top_k", 20),
        }
        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            try:
                gen_out = self._model.generate(
                    **inputs, **base_kwargs, **sampling_kwargs
                )
            except TypeError:
                gen_out = self._model.generate(**inputs, **base_kwargs)

        text_ids = gen_out[0] if isinstance(gen_out, (tuple, list)) else gen_out
        seq = getattr(text_ids, "sequences", text_ids)
        return self._processor.batch_decode(
            seq[:, prompt_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    def _generate_transformers_batch(self, messages_list: List[list]) -> List[str]:
        inputs = self._prep_transformers_inputs(messages_list)
        return self._run_transformers_generate(inputs)


# ---------------------------------------------------------------------------
# LLM client for verify / rewrite on Qwen3-Omni (duck-typed to BaseLLMClient)
# ---------------------------------------------------------------------------
class QwenOmniLLMClient:
    """A ``BaseLLMClient``-compatible client backed by a Qwen3-Omni engine.

    ``video_uri`` is LOCAL clip path(s) — no Files API upload: ``None`` (text
    only), a single path, or a list of paths. Returns parsed JSON (markdown
    fences / surrounding prose tolerated). Duck-typed (not a subclass) so this
    module never imports google-genai.
    """

    def __init__(self, engine: "Qwen3OmniCaptioner", max_retries: int = 2) -> None:
        self._engine = engine
        self.max_retries = max(1, max_retries)

    @staticmethod
    def _clip_paths(video_uri) -> List[str]:
        if video_uri is None:
            return []
        if isinstance(video_uri, str):
            return [video_uri]
        return list(video_uri)

    def generate_json(
        self, prompt: str, schema_name: str, video_uri=None
    ) -> Dict[str, Any]:
        messages = Qwen3OmniCaptioner._build_messages_multi(
            prompt, self._clip_paths(video_uri)
        )
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            raw = self._engine.generate(messages)
            try:
                return extract_caption_json(raw)
            except CaptionParseError as e:
                last_error = e
                print(f"  [attempt {attempt}/{self.max_retries}] Qwen JSON parse "
                      f"error for {schema_name}: {e}. Retrying...")
        raise RuntimeError(
            f"Qwen3-Omni call failed after {self.max_retries} attempts "
            f"(schema={schema_name}): {last_error}"
        )

    def generate_json_many(
        self, prompts: List[str], schema_name: str, video_uris=None
    ) -> List[Dict[str, Any]]:
        """Run N prompts in ONE batched model forward (each keeps its own clip(s)).

        An item whose output won't parse falls back to a single retried
        ``generate_json`` so one bad query never fails the batch.
        """
        if not prompts:
            return []
        uris = list(video_uris) if video_uris else [None] * len(prompts)
        messages_list = [
            Qwen3OmniCaptioner._build_messages_multi(p, self._clip_paths(u))
            for p, u in zip(prompts, uris)
        ]
        raw_texts = self._engine.generate_many(messages_list)
        out: List[Dict[str, Any]] = []
        for prompt, uri, raw in zip(prompts, uris, raw_texts):
            try:
                out.append(extract_caption_json(raw))
            except CaptionParseError:
                out.append(self.generate_json(prompt, schema_name, video_uri=uri))
        return out

    def usage_report(self) -> Dict[str, Any]:
        """Interface parity with GeminiLLMClient (local model: no token counts)."""
        empty = {
            "calls": 0, "prompt_tokens": 0,
            "candidates_tokens": 0, "total_tokens": 0,
        }
        return {"by_stage": {}, "total": dict(empty)}
