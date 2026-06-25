"""Dual observation-caption backend: Qwen3-VL (frames) + TimeChat (clip).

OBSERVATION-ONLY. Neither model emits emotion — emotion is decided later by the
Gemini emotion-event stage. Per segment:

- ``Qwen3VLCaptioner`` reads N sampled frames and produces the VISUAL fields
  (``visual_objective`` + ``visual_expression`` + confidence/evidence_strength).
- ``TimeChatCaptioner`` reads the segment CLIP (audio+video) and produces the
  AUDIO/TEMPORAL fields (``audio_description`` + ``temporal_description``).

The two are merged by ``segment_id`` into one observation ``OmniCaption`` and
cached. The whole cache / resume / salvage path is reused from ``caption_utils``
so a missing/garbled half never drops a segment.

Both models load lazily (heavy imports + weights only on first inference), mirror
``Qwen3OmniCaptioner``. They cannot be exercised on a machine without a GPU; the
exact model classes / processors / prompt format are flagged with NOTE where they
may need adjustment on the inference server.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .caption_utils import (
    CaptionParseError,
    atomic_write_json,
    caption_cache_path,
    extract_caption_json,
    missing_required_fields,
    salvage_caption,
    save_raw_output,
    _chunks,
    _resolve_cache,
)
from .clip_extractor import extract_frames
from .io_utils import load_prompt_template
from .models import OmniCaption, Segment

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

DEFAULT_QWEN3VL_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_TIMECHAT_MODEL = "yaolily/TimeChat-Captioner-GRPO-7B"
DEFAULT_FRAMES_PER_SEGMENT = 5

# Per-image pixel cap for Qwen3-VL frames (keeps vision-token count / prefill in
# check; ~ the value from the TimeChat reference example).
VL_MAX_PIXELS = 297920

# TimeChat video-sampling params (from the official TimeChat example): it reads
# the clip directly at fps with a per-frame pixel cap.
TIMECHAT_MAX_PIXELS = 297920
TIMECHAT_FPS = 2.0
TIMECHAT_MAX_FRAMES = 160


def _free_gpu() -> None:
    """Release GPU memory after dropping a model's references (best-effort).

    Prints the post-cleanup allocated/reserved memory so it is VISIBLE whether the
    weights actually left the GPU (rather than guessing).
    """
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            alloc = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"[gpu] after cleanup: allocated={alloc:.1f}GB "
                  f"reserved={reserved:.1f}GB")
    except Exception:
        pass

DEFAULT_SAMPLING_PARAMS: Dict[str, Any] = {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "max_tokens": 1024,
}


# ---------------------------------------------------------------------------
# Qwen3-VL backend (frames -> visual observation)
# ---------------------------------------------------------------------------
@dataclass
class Qwen3VLCaptioner:
    """Qwen3-VL over sampled frames. Loads on first use; import-safe to construct."""

    model_path: str = DEFAULT_QWEN3VL_MODEL
    sampling_params: Dict[str, Any] = field(
        default_factory=lambda: dict(DEFAULT_SAMPLING_PARAMS)
    )
    device_map: str = "auto"
    attn_implementation: Optional[str] = None

    _model: Any = field(default=None, init=False, repr=False)
    _processor: Any = field(default=None, init=False, repr=False)

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        import time as _time
        # NOTE: model class / processor names per the Qwen3-VL transformers release.
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        t0 = _time.perf_counter()
        print(f"[model] loading {self.model_path} (Qwen3-VL)...")
        load_kwargs: Dict[str, Any] = {"dtype": "auto", "device_map": self.device_map}
        if self.attn_implementation:
            load_kwargs["attn_implementation"] = self.attn_implementation
        self._model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_path, **load_kwargs
        )
        self._processor = AutoProcessor.from_pretrained(self.model_path)
        print(f"[model] loaded {self.model_path} in {_time.perf_counter() - t0:.1f}s")

    @staticmethod
    def _build_messages_frames(prompt_text: str, frame_paths: List[str]) -> list:
        content = [
            {"type": "image", "image": p, "max_pixels": VL_MAX_PIXELS}
            for p in frame_paths if p
        ]
        content.append({"type": "text", "text": prompt_text})
        return [{"role": "user", "content": content}]

    def generate(self, messages: list) -> str:
        out = self.generate_many([messages])
        return out[0] if out else ""

    def generate_many(self, messages_list: List[list]) -> List[str]:
        if not messages_list:
            return []
        self._ensure_model()
        import torch
        from qwen_vl_utils import process_vision_info  # lazy

        texts = [
            self._processor.apply_chat_template(
                m, tokenize=False, add_generation_prompt=True
            )
            for m in messages_list
        ]
        image_inputs, video_inputs = process_vision_info(messages_list)
        inputs = self._processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self._model.device)
        sp = self.sampling_params
        with torch.no_grad():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=sp.get("max_tokens", 1024),
                do_sample=sp.get("temperature", 0.0) > 0,
                temperature=sp.get("temperature", 0.6),
                top_p=sp.get("top_p", 0.95),
                top_k=sp.get("top_k", 20),
            )
        trimmed = [
            out[len(inp):] for inp, out in zip(inputs.input_ids, generated)
        ]
        return self._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    def unload(self) -> None:
        """Drop the model + processor and free GPU memory (for phased loading)."""
        if self._model is not None:
            print(f"[model] unloading {self.model_path} (Qwen3-VL)")
        self._model = None
        self._processor = None
        _free_gpu()


# ---------------------------------------------------------------------------
# TimeChat backend (clip -> audio/temporal observation)
# ---------------------------------------------------------------------------
@dataclass
class TimeChatCaptioner:
    """TimeChat-Captioner (qwen2_5_omni arch) over the segment clip. Lazy load."""

    model_path: str = DEFAULT_TIMECHAT_MODEL
    sampling_params: Dict[str, Any] = field(
        default_factory=lambda: dict(DEFAULT_SAMPLING_PARAMS)
    )
    use_audio_in_video: bool = True
    device_map: str = "auto"
    attn_implementation: Optional[str] = None
    video_reader_backend: str = "torchvision"

    _model: Any = field(default=None, init=False, repr=False)
    _processor: Any = field(default=None, init=False, repr=False)

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        import time as _time
        if self.video_reader_backend:
            os.environ["FORCE_QWENVL_VIDEO_READER"] = self.video_reader_backend
        # NOTE: TimeChat is the qwen2_5_omni architecture (base Qwen2.5-Omni-7B).
        from transformers import (
            Qwen2_5OmniForConditionalGeneration,
            Qwen2_5OmniProcessor,
        )

        t0 = _time.perf_counter()
        print(f"[model] loading {self.model_path} (TimeChat / qwen2_5_omni)...")
        load_kwargs: Dict[str, Any] = {"dtype": "auto", "device_map": self.device_map}
        if self.attn_implementation:
            load_kwargs["attn_implementation"] = self.attn_implementation
        self._model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            self.model_path, **load_kwargs
        )
        # Text-only captioning: drop the audio-generating talker if supported.
        if hasattr(self._model, "disable_talker"):
            try:
                self._model.disable_talker()
            except Exception:
                pass
        self._processor = Qwen2_5OmniProcessor.from_pretrained(self.model_path)
        print(f"[model] loaded {self.model_path} in {_time.perf_counter() - t0:.1f}s")

    @staticmethod
    def _build_messages_clip(prompt_text: str, clip_path: str) -> list:
        # Text first, then the video with the reference sampling params (TimeChat
        # reads the clip directly at fps with a per-frame pixel cap).
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt_text}]
        if clip_path:
            content.append({
                "type": "video",
                "video": clip_path,
                "max_pixels": TIMECHAT_MAX_PIXELS,
                "max_frames": TIMECHAT_MAX_FRAMES,
                "fps": TIMECHAT_FPS,
                "video_max_pixels": TIMECHAT_MAX_PIXELS,
            })
        return [{"role": "user", "content": content}]

    def generate(self, messages: list) -> str:
        out = self.generate_many([messages])
        return out[0] if out else ""

    def generate_many(self, messages_list: List[list]) -> List[str]:
        if not messages_list:
            return []
        self._ensure_model()
        import torch
        from qwen_omni_utils import process_mm_info  # lazy

        text = self._processor.apply_chat_template(
            messages_list, add_generation_prompt=True, tokenize=False
        )
        audios, images, videos = process_mm_info(
            messages_list, use_audio_in_video=self.use_audio_in_video
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
        inputs = inputs.to(self._model.device).to(self._model.dtype)
        sp = self.sampling_params
        prompt_len = inputs["input_ids"].shape[1]
        # Qwen2.5-Omni (TimeChat) generate kwargs: talker off + thinker token cap.
        # NOTE: some builds reject TimeChat-specific kwargs like ``in_video`` (they
        # raise ValueError "model_kwargs are not used"); fall back to plain
        # max_new_tokens in that case.
        gen_kwargs: Dict[str, Any] = {
            "return_audio": False,
            "thinker_max_new_tokens": sp.get("max_tokens", 1024),
        }
        with torch.no_grad():
            try:
                gen_out = self._model.generate(**inputs, **gen_kwargs)
            except (TypeError, ValueError):
                gen_out = self._model.generate(
                    **inputs, return_audio=False,
                    max_new_tokens=sp.get("max_tokens", 1024),
                )
        text_ids = gen_out[0] if isinstance(gen_out, (tuple, list)) else gen_out
        seq = getattr(text_ids, "sequences", text_ids)
        return self._processor.batch_decode(
            seq[:, prompt_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    def unload(self) -> None:
        """Drop the model + processor and free GPU memory (for phased loading)."""
        if self._model is not None:
            print(f"[model] unloading {self.model_path} (TimeChat)")
        self._model = None
        self._processor = None
        _free_gpu()


# ---------------------------------------------------------------------------
# Merge + per-video captioning (with resume)
# ---------------------------------------------------------------------------
def _merge_strict(
    visual_raw: str, audio_raw: str, segment: Segment, video_id: str
) -> Optional[OmniCaption]:
    """Merge the VL JSON + TimeChat free-form text into a valid OmniCaption.

    Visual fields come from the VL JSON object. TimeChat is a free-form captioner
    (NOT JSON), so its raw text is stored directly as ``audio_description``.
    Metadata is forced from the trusted segment. Returns None if the VL half can't
    be parsed or the merged caption is incomplete/invalid (caller salvages).
    """
    try:
        vobj = extract_caption_json(visual_raw)
    except CaptionParseError:
        return None
    data: Dict[str, Any] = dict(vobj) if isinstance(vobj, dict) else {}
    audio_text = (audio_raw or "").strip()
    data["audio_description"] = audio_text or "no audible cue"
    data.setdefault("temporal_description", "")
    data["segment_id"] = segment.segment_id
    data["time_range"] = [round(segment.start_time, 2), round(segment.end_time, 2)]
    data["video_id"] = video_id
    if missing_required_fields(data):
        return None
    try:
        return OmniCaption.model_validate(data)
    except Exception:
        return None


def _salvage_merge(
    visual_raw: str, audio_raw: str, segment: Segment, video_id: str
) -> OmniCaption:
    """Best-effort merge keeping whatever decoded (never drops the segment)."""
    item: Dict[str, Any] = {}
    try:
        v = extract_caption_json(visual_raw)
        if isinstance(v, dict):
            item.update(v)
    except CaptionParseError:
        pass
    audio_text = (audio_raw or "").strip()
    if audio_text:
        item["audio_description"] = audio_text
    combined = f"--- VISUAL ---\n{visual_raw}\n--- TIMECHAT ---\n{audio_raw}"
    return salvage_caption(item or None, combined, segment, video_id)


def caption_video_observation(
    video_id: str,
    segments: List[Segment],
    vl_captioner: Qwen3VLCaptioner,
    tc_captioner: TimeChatCaptioner,
    cache_dir: Path,
    raw_dir: Path,
    resume: bool = True,
    overwrite: bool = False,
    frames_per_segment: int = DEFAULT_FRAMES_PER_SEGMENT,
    parallel: int = 1,
    prompts_dir: Optional[Path] = None,
) -> List[OmniCaption]:
    """Caption every segment of one video with the dual VL+TimeChat backend.

    Resume is applied first (cached segments skipped). Cache-miss segments are
    grouped ``parallel`` at a time; each group runs one batched VL call and one
    batched TimeChat call, the halves are merged per segment, valid captions are
    cached, and a missing/garbled half is salvaged (uncached, retried next run).
    """
    results_by_id: Dict[str, OmniCaption] = {}
    cached, to_generate = _resolve_cache(
        video_id, segments, cache_dir, resume, overwrite
    )
    results_by_id.update(cached)

    visual_prompt = load_prompt_template(
        prompts_dir or _PROMPTS_DIR, "observation_visual_prompt.txt"
    )
    audio_prompt = load_prompt_template(
        prompts_dir or _PROMPTS_DIR, "observation_audio_temporal_prompt.txt"
    )

    for group in _chunks(to_generate, max(1, parallel)):
        ids = ", ".join(s.segment_id for s in group)
        print(f"[caption] generate(observation): video_id={video_id} "
              f"segment_ids=[{ids}] (frames/seg={frames_per_segment})")
        vl_messages = []
        tc_messages = []
        for seg in group:
            try:
                frames = extract_frames(seg.clip_path, frames_per_segment)
            except Exception as e:
                print(f"[caption] frame extract failed {seg.segment_id}: {e}")
                frames = []
            vl_messages.append(
                Qwen3VLCaptioner._build_messages_frames(visual_prompt, frames)
            )
            tc_messages.append(
                TimeChatCaptioner._build_messages_clip(audio_prompt, seg.clip_path or "")
            )
        try:
            vl_raws = vl_captioner.generate_many(vl_messages)
        except Exception as e:
            print(f"[caption] Qwen3-VL generate failed for {len(group)}: {e}")
            vl_raws = [""] * len(group)
        try:
            tc_raws = tc_captioner.generate_many(tc_messages)
        except Exception as e:
            print(f"[caption] TimeChat generate failed for {len(group)}: {e}")
            tc_raws = [""] * len(group)

        for seg, vraw, traw in zip(group, vl_raws, tc_raws):
            cap = _merge_strict(vraw, traw, seg, video_id)
            if cap is not None:
                atomic_write_json(
                    caption_cache_path(cache_dir, video_id, seg.segment_id),
                    cap.model_dump(),
                )
                results_by_id[seg.segment_id] = cap
                continue
            combined = f"--- VISUAL ---\n{vraw}\n--- AUDIO/TEMPORAL ---\n{traw}"
            raw_path = save_raw_output(
                raw_dir, video_id, seg.segment_id, combined, "salvaged_observation"
            )
            results_by_id[seg.segment_id] = _salvage_merge(vraw, traw, seg, video_id)
            print(f"[caption] salvaged observation (uncached): "
                  f"video_id={video_id} segment_id={seg.segment_id} "
                  f"raw_saved={raw_path}")

    return [
        results_by_id[s.segment_id]
        for s in segments
        if s.segment_id in results_by_id
    ]
