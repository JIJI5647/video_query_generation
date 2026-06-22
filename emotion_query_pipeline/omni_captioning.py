"""Qwen3-Omni structured captioning backend — one segment per prompt.

Design / safety constraints (this machine cannot run Qwen3-Omni):

- LAZY EVERYTHING. ``vllm``, ``torch``, ``transformers`` and ``qwen_omni_utils``
  are imported, and the 30B model is loaded, only inside ``Qwen3OmniCaptioner._
  ensure_model`` / ``.caption`` — never at module import. Importing this module,
  constructing the captioner, building prompts, extracting/validating JSON, and
  the whole cache/resume path are pure Python and run without the heavy deps or
  any GPU. So mock tests, parser tests, prompt-construction tests and
  cache/resume tests all work locally.

- ONE SEGMENT PER PROMPT. Each model call captions exactly one segment's clip
  and returns one structured ``OmniCaption``. There is no multi-segment batching
  (``caption_batch_size`` is 1), which prevents segment_id / time_range / caption
  mismatch.

- RESUME / CACHE. Each caption is written atomically to
  ``<cache_dir>/<video_id>/<segment_id>.json``. On rerun, a segment whose cache
  parses and has all required fields is skipped; an invalid/missing cache is
  regenerated. Raw model text for a parse failure is saved to
  ``<raw_dir>/<video_id>/<segment_id>.txt`` for debugging (never silently
  dropped).

Downstream generation / verification / export are unchanged: the rich
``OmniCaption`` is adapted to the existing flat ``EmotionCaption`` via
``omni_to_emotion_caption`` (the compatibility / field-mapping layer).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

from .io_utils import load_prompt_template
from .models import (
    EMOTION_LABEL_VALUES,
    EmotionCaption,
    OMNI_REQUIRED_FIELDS,
    OmniCaption,
    Segment,
)

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

DEFAULT_MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

# Sampling defaults (spec §4.5). Plain dict so the SamplingParams object is only
# built lazily inside the captioner (vllm import stays out of module load).
DEFAULT_SAMPLING_PARAMS: Dict[str, Any] = {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "max_tokens": 2048,
}

# Default per-prompt caption batch size. Qwen3-Omni captioning is strictly one
# segment per prompt; this is exposed only so callers can assert/log it.
DEFAULT_CAPTION_BATCH_SIZE = 1


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class CaptionParseError(Exception):
    """Raised when a raw model output can't be turned into a valid OmniCaption.

    ``reason`` is a short machine code used in logs / cache-invalidation
    messages: ``json_parse_error``, ``missing_required_fields`` or
    ``schema_validation_error``. ``raw_text`` is the offending model output (for
    debugging / raw dump).
    """

    def __init__(self, reason: str, message: str, raw_text: str = "") -> None:
        super().__init__(f"{reason}: {message}")
        self.reason = reason
        self.raw_text = raw_text


# ---------------------------------------------------------------------------
# Backend protocol (so tests can inject a fake without any heavy deps)
# ---------------------------------------------------------------------------
class CaptionerProtocol(Protocol):
    """Anything that turns (prompt_text, clip_path) into raw model text."""

    def caption(self, prompt_text: str, clip_path: str) -> str:  # pragma: no cover
        ...


# ---------------------------------------------------------------------------
# Prompt construction (one segment per prompt)
# ---------------------------------------------------------------------------
def build_omni_caption_prompt(
    segment: Segment, prompts_dir: Optional[Path] = None
) -> str:
    """Build the structured caption prompt for exactly ONE segment's clip."""
    template = load_prompt_template(
        prompts_dir or _PROMPTS_DIR, "omni_caption_prompt.txt"
    )
    prompt = template
    prompt = prompt.replace("{segment_id}", segment.segment_id)
    prompt = prompt.replace("{start_time}", f"{segment.start_time:.2f}")
    prompt = prompt.replace("{end_time}", f"{segment.end_time:.2f}")
    return prompt


# ---------------------------------------------------------------------------
# Robust JSON extraction + validation + parsing
# ---------------------------------------------------------------------------
def extract_caption_json(raw_text: str) -> dict:
    """Pull the first JSON object out of a raw model response.

    Tolerates markdown fences (```json ... ```), leading/trailing prose, and a
    second trailing JSON value. Raises ``CaptionParseError(json_parse_error)``
    if no JSON object can be decoded.
    """
    if raw_text is None:
        raise CaptionParseError("json_parse_error", "empty (None) response", "")
    text = raw_text.strip()
    if not text:
        raise CaptionParseError("json_parse_error", "empty response", raw_text)

    # Strip a fenced block if present (```json ... ``` or ``` ... ```).
    if "```" in text:
        fence = text.find("```")
        rest = text[fence + 3 :]
        if "\n" in rest:
            rest = rest.split("\n", 1)[1]  # drop the ```json language tag line
        end = rest.rfind("```")
        if end != -1:
            rest = rest[:end]
        text = rest.strip()

    # Locate the first '{' and decode one JSON value, ignoring trailing junk.
    start = text.find("{")
    if start == -1:
        raise CaptionParseError(
            "json_parse_error", "no JSON object found in output", raw_text
        )
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError as e:
        raise CaptionParseError("json_parse_error", str(e), raw_text) from e
    if not isinstance(obj, dict):
        raise CaptionParseError(
            "json_parse_error", "top-level JSON is not an object", raw_text
        )
    return obj


def missing_required_fields(data: dict) -> List[str]:
    """Required top-level fields (spec §9.2) that are absent or empty.

    ``time_range`` must be a 2-element list; the other required fields just have
    to be present and non-None (empty strings/lists are allowed for the content
    fields so a genuinely sparse-but-complete caption still validates).
    """
    missing: List[str] = []
    for fld in OMNI_REQUIRED_FIELDS:
        if fld not in data or data[fld] is None:
            missing.append(fld)
            continue
        if fld == "time_range":
            tr = data[fld]
            if not isinstance(tr, (list, tuple)) or len(tr) != 2:
                missing.append(fld)
    return missing


def parse_caption(raw_text: str, segment: Segment, video_id: str) -> OmniCaption:
    """Extract -> validate required fields -> build a metadata-corrected OmniCaption.

    ``segment_id`` / ``time_range`` / ``video_id`` are forced from the segment so
    the cached metadata is always correct even if the model echoed them wrong.
    Raises ``CaptionParseError`` (with ``raw_text``) on any failure.
    """
    data = extract_caption_json(raw_text)
    missing = missing_required_fields(data)
    if missing:
        raise CaptionParseError(
            "missing_required_fields",
            f"missing/invalid: {', '.join(missing)}",
            raw_text,
        )
    # Overwrite metadata from the trusted segment (never trust model echo).
    data["segment_id"] = segment.segment_id
    data["time_range"] = [round(segment.start_time, 2), round(segment.end_time, 2)]
    data["video_id"] = video_id
    try:
        return OmniCaption.model_validate(data)
    except Exception as e:  # pydantic ValidationError or anything odd in subfields
        raise CaptionParseError("schema_validation_error", str(e), raw_text) from e


# ---------------------------------------------------------------------------
# Cache / resume helpers (atomic write, raw dump)
# ---------------------------------------------------------------------------
def caption_cache_path(cache_dir: Path, video_id: str, segment_id: str) -> Path:
    return Path(cache_dir) / video_id / f"{segment_id}.json"


def raw_output_path(raw_dir: Path, video_id: str, segment_id: str) -> Path:
    return Path(raw_dir) / video_id / f"{segment_id}.txt"


def atomic_write_json(path: Path, data: dict) -> None:
    """Write ``data`` as JSON to ``path`` atomically (tmp file + flush + rename).

    Avoids leaving a half-written ``<segment_id>.json`` if the process is killed
    mid-write.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)  # atomic on POSIX/Windows for same-filesystem rename


def save_raw_output(
    raw_dir: Path, video_id: str, segment_id: str, raw_text: str, reason: str
) -> Path:
    """Persist a failed model output for debugging. Returns the file path."""
    path = raw_output_path(raw_dir, video_id, segment_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = f"# parse failure: reason={reason}\n# segment_id={segment_id}\n\n"
    path.write_text(header + (raw_text or ""), encoding="utf-8")
    return path


def read_valid_cache(path: Path):
    """Load a cached caption if it is present, parseable and complete.

    Returns ``(OmniCaption, None)`` on a cache hit, or ``(None, reason)`` where
    ``reason`` is ``"not_found"``, ``"json_parse_error"``,
    ``"missing_required_fields"`` or ``"schema_validation_error"``.
    """
    path = Path(path)
    if not path.exists():
        return None, "not_found"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None, "json_parse_error"
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None, "json_parse_error"
    if not isinstance(data, dict):
        return None, "json_parse_error"
    missing = missing_required_fields(data)
    if missing:
        return None, "missing_required_fields"
    try:
        return OmniCaption.model_validate(data), None
    except Exception:
        return None, "schema_validation_error"


# ---------------------------------------------------------------------------
# Compatibility layer: OmniCaption -> EmotionCaption
# ---------------------------------------------------------------------------
def _label_from_description(emotion_description: str) -> str:
    """Best-effort map a free-text emotion reading to one fixed label.

    The structured caption carries a NL ``emotion_description`` rather than a
    fixed label; the existing generation/filter path keys off the eight
    ``EMOTION_LABEL_VALUES``. We scan for one of those words (longest first so
    e.g. "disappointed" wins over a substring) and fall back to ``"neutral"``
    when none is clearly present (filtered out later, like a neutral clip).
    """
    text = (emotion_description or "").lower()
    for label in sorted(EMOTION_LABEL_VALUES, key=len, reverse=True):
        if label in text:
            return label
    return "neutral"


def _flatten(items: List[Any]) -> List[str]:
    out: List[str] = []
    for it in items or []:
        if it is None:
            continue
        s = it if isinstance(it, str) else str(it)
        s = s.strip()
        if s:
            out.append(s)
    return out


def omni_to_emotion_caption(oc: OmniCaption, video_id: str) -> EmotionCaption:
    """Adapt a structured OmniCaption to the flat EmotionCaption the rest of the
    pipeline (generation / filter / export / stats) already consumes.

    ``segment_ids`` stays a single-element list (the segment's id), so internal
    segment_id -> clip mapping used by verification is preserved.
    """
    vo = oc.visual_objective
    persons = _flatten([p.person for p in vo.people])
    person = "; ".join(persons) if persons else "not described"

    actions = _flatten([p.action for p in vo.people]) + _flatten(vo.key_actions)
    action = "; ".join(actions) if actions else "not described"

    evidence: List[str] = []
    for ve in oc.visual_expression:
        evidence.extend(_flatten(ve.facial_cues))
        evidence.extend(_flatten(ve.body_cues))
        if ve.gaze and ve.gaze.strip():
            evidence.append(ve.gaze.strip())

    sound = oc.audio_description.strip() if oc.audio_description else ""
    if not sound:
        sound = "no audible cue"

    return EmotionCaption(
        video_id=video_id,
        caption_id=f"{video_id}_{oc.segment_id}",
        segment_ids=[oc.segment_id],
        person=person,
        action=action,
        sound=sound,
        emotion=_label_from_description(oc.emotion_description),
        confidence=oc.confidence,
        evidence_strength=oc.evidence_strength,
        observable_evidence=evidence,
    )


# ---------------------------------------------------------------------------
# Per-segment + per-video captioning (with resume)
# ---------------------------------------------------------------------------
def caption_one_segment(
    captioner: CaptionerProtocol,
    video_id: str,
    segment: Segment,
    cache_dir: Path,
    raw_dir: Path,
    resume: bool = True,
    overwrite: bool = False,
    prompts_dir: Optional[Path] = None,
) -> OmniCaption:
    """Caption one segment, honouring the resume cache.

    Resume hit (``resume`` and not ``overwrite`` and a valid cache file) returns
    the cached caption WITHOUT calling the model. Otherwise the model is called
    once for this single segment, the result parsed/validated and written
    atomically. Parse failures save the raw output and re-raise
    ``CaptionParseError``.
    """
    cache_path = caption_cache_path(cache_dir, video_id, segment.segment_id)

    if resume and not overwrite:
        cached, reason = read_valid_cache(cache_path)
        if cached is not None:
            print(f"[caption] skip existing: video_id={video_id} "
                  f"segment_id={segment.segment_id}")
            return cached
        if reason != "not_found":
            print(f"[caption] regenerate invalid cache: video_id={video_id} "
                  f"segment_id={segment.segment_id} reason={reason}")

    print(f"[caption] generate: video_id={video_id} "
          f"segment_id={segment.segment_id}")
    prompt = build_omni_caption_prompt(segment, prompts_dir)
    raw_text = captioner.caption(prompt, segment.clip_path or "")
    try:
        caption = parse_caption(raw_text, segment, video_id)
    except CaptionParseError as e:
        raw_path = save_raw_output(
            raw_dir, video_id, segment.segment_id, e.raw_text, e.reason
        )
        print(f"[caption] parse failed: video_id={video_id} "
              f"segment_id={segment.segment_id} reason={e.reason} "
              f"raw_saved={raw_path}")
        raise
    atomic_write_json(cache_path, caption.model_dump())
    return caption


def caption_video_omni(
    video_id: str,
    segments: List[Segment],
    captioner: CaptionerProtocol,
    cache_dir: Path,
    raw_dir: Path,
    resume: bool = True,
    overwrite: bool = False,
    caption_batch_size: int = DEFAULT_CAPTION_BATCH_SIZE,
    prompts_dir: Optional[Path] = None,
) -> List[OmniCaption]:
    """Caption every segment of one video, one segment per model call.

    ``caption_batch_size`` must be 1 (one segment per prompt is enforced); a
    larger value is clamped with a warning. A per-segment parse failure is logged
    (with raw output saved) and that segment is skipped — it never aborts the
    whole video.
    """
    if caption_batch_size != 1:
        print(f"[caption] WARNING: caption_batch_size={caption_batch_size} "
              f"ignored; Qwen3-Omni captioning is one segment per prompt.")

    captions: List[OmniCaption] = []
    for segment in segments:
        if not segment.clip_path:
            continue
        try:
            captions.append(
                caption_one_segment(
                    captioner, video_id, segment, cache_dir, raw_dir,
                    resume=resume, overwrite=overwrite, prompts_dir=prompts_dir,
                )
            )
        except CaptionParseError:
            continue  # already logged + raw saved; skip this segment
    return captions


# ---------------------------------------------------------------------------
# The actual Qwen3-Omni backend (lazy model load — never at import time)
# ---------------------------------------------------------------------------
@dataclass
class Qwen3OmniCaptioner:
    """vLLM-backed Qwen3-Omni captioner. The model loads on first ``caption``.

    Constructing this object is cheap and import-safe: it stores config only. No
    ``vllm`` / ``torch`` / ``transformers`` / ``qwen_omni_utils`` import happens
    until ``_ensure_model`` runs (inside the first ``caption`` call), so nothing
    here touches a GPU or downloads weights on this machine.
    """

    model_path: str = DEFAULT_MODEL_PATH
    sampling_params: Dict[str, Any] = field(
        default_factory=lambda: dict(DEFAULT_SAMPLING_PARAMS)
    )
    use_audio_in_video: bool = True
    gpu_memory_utilization: float = 0.95
    max_num_seqs: int = 8
    max_model_len: int = 32768
    tensor_parallel_size: Optional[int] = None
    seed: int = 1234
    limit_mm_per_prompt: Optional[Dict[str, int]] = None

    _llm: Any = field(default=None, init=False, repr=False)
    _processor: Any = field(default=None, init=False, repr=False)
    _sampling: Any = field(default=None, init=False, repr=False)

    def _ensure_model(self) -> None:
        """Load the model + processor exactly once. Heavy imports live here."""
        if self._llm is not None:
            return
        # Lazy imports — keep these OFF the module import path.
        import torch
        from vllm import LLM, SamplingParams
        from transformers import Qwen3OmniMoeProcessor

        os.environ.setdefault("VLLM_USE_V1", "0")
        tp = self.tensor_parallel_size or max(1, torch.cuda.device_count())
        limit_mm = self.limit_mm_per_prompt or {"image": 3, "video": 3, "audio": 3}

        self._llm = LLM(
            model=self.model_path,
            trust_remote_code=True,
            gpu_memory_utilization=self.gpu_memory_utilization,
            tensor_parallel_size=tp,
            limit_mm_per_prompt=limit_mm,
            max_num_seqs=self.max_num_seqs,
            max_model_len=self.max_model_len,
            seed=self.seed,
        )
        self._sampling = SamplingParams(**self.sampling_params)
        self._processor = Qwen3OmniMoeProcessor.from_pretrained(self.model_path)

    @staticmethod
    def _build_messages(prompt_text: str, clip_path: str) -> list:
        """One user turn: the segment video (audio included) then the text prompt."""
        return [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": clip_path},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

    def caption(self, prompt_text: str, clip_path: str) -> str:
        """Run one Qwen3-Omni inference on a single segment clip; return raw text."""
        self._ensure_model()
        from qwen_omni_utils import process_mm_info  # lazy

        messages = self._build_messages(prompt_text, clip_path)
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        audios, images, videos = process_mm_info(
            messages, use_audio_in_video=self.use_audio_in_video
        )
        inputs: Dict[str, Any] = {
            "prompt": text,
            "multi_modal_data": {},
            "mm_processor_kwargs": {"use_audio_in_video": self.use_audio_in_video},
        }
        if images is not None:
            inputs["multi_modal_data"]["image"] = images
        if videos is not None:
            inputs["multi_modal_data"]["video"] = videos
        if audios is not None:
            inputs["multi_modal_data"]["audio"] = audios

        outputs = self._llm.generate([inputs], sampling_params=self._sampling)
        return outputs[0].outputs[0].text
