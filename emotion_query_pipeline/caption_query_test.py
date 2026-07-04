"""Caption-model → normalized caption → existing Gemini downstream → queries.

This module is the reusable core behind ``run_caption_query_test.py``. Its goal is
NOT to benchmark a caption model in isolation, but to verify that a **new caption
model's output can be fed through the EXISTING pipeline** (the Gemini
emotion-event stage + query-generation stage) and produce real queries.

Design constraints (mirroring ``omni_captioning.py``):

- IMPORT-SAFE / LAZY EVERYTHING. Importing this module must not pull in
  ``torch`` / ``transformers`` / ``qwen_omni_utils`` / ``qwen_vl_utils`` /
  ``decord`` / ``soundfile`` or any model repo, must not download weights, need a
  GPU, or need Gemini credentials. Every heavy dependency is imported *inside* the
  model-runner branch that needs it. All of normalization, boundary validation,
  the downstream builders and the output writers are pure Python.

- NORMALIZE INTO THE EXISTING SCHEMA. Whatever a caption model emits (plain text,
  JSON, a dict, or malformed output) is coerced into ``models.OmniCaption`` — the
  same structure the Gemini downstream already consumes — with the trusted
  metadata (``video_id`` / ``segment_id`` / ``time_range``) forced from the
  ``Segment`` and never taken from the model.

The heavy runners for the uncertain model repos (AVoCaDO, TimeChat, Audio
Flamingo 3, SECap) raise a clear ``NotImplementedError`` with setup instructions
rather than guessing a broken API.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .io_utils import load_prompt_template, write_jsonl
from .models import OmniCaption, Segment

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

_CONFIDENCE_VALUES = ("high", "medium", "low")
_EVIDENCE_VALUES = ("clear", "ambiguous", "weak")

# Text-caption instructions for the single-modality sub-runners (observation
# only, no emotion — emotion is judged later by the Gemini emotion-event stage).
VIDEO_CAPTION_INSTRUCTION = (
    "Watch this short video clip (no audio). In 2-4 sentences describe ONLY what "
    "is visible: the people (appearance, position, visibility), their observable "
    "actions, the scene, and any observable facial cues, body cues, posture, "
    "gestures and gaze. Do NOT name or infer any emotion, and do NOT describe "
    "sound or speech. Describe only what is visible."
)
AUDIO_CAPTION_INSTRUCTION = (
    "Listen to this short audio clip. In 1-3 sentences describe HOW the audio "
    "sounds (voice quality, prosody, non-speech sounds such as shouting, crying, "
    "laughing, heavy breathing, silence) — not the exact spoken words. Do NOT "
    "describe anything visual."
)


# ---------------------------------------------------------------------------
# Supported caption models (metadata registry)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelSpec:
    """Static description of one supported caption model / model-pair.

    ``kind`` is ``"av"`` (a single audio-visual model consuming the video clip) or
    ``"audio_video"`` (a separate audio model + a separate video model whose text
    outputs are merged). ``requires_video`` / ``requires_audio`` drive the input
    boundary checks.
    """

    name: str
    kind: str  # "av" | "audio_video"
    requires_video: bool
    requires_audio: bool
    default_model_path: str = ""
    default_audio_model_path: str = ""
    default_video_model_path: str = ""
    non_commercial: bool = False
    note: str = ""


# NOTE: for ``audio_video`` models the audio evidence comes from the audio model
# and the visual evidence from the video model; they are merged so an audio-only
# model never fabricates visual evidence and a video-only model never fabricates
# audio evidence.
CAPTION_MODEL_SPECS: Dict[str, ModelSpec] = {
    "avocado": ModelSpec(
        name="avocado",
        kind="av",
        requires_video=True,
        requires_audio=False,
        default_model_path="AVoCaDO-Captioner/AVoCaDO",
        note="AV caption model.",
    ),
    "qwen3_omni": ModelSpec(
        name="qwen3_omni",
        kind="av",
        requires_video=True,
        requires_audio=False,
        default_model_path="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        note="Omni AV baseline (reuses the pipeline's Qwen3-Omni captioner).",
    ),
    "timechat": ModelSpec(
        name="timechat",
        kind="av",
        requires_video=True,
        requires_audio=False,
        default_model_path="yaolily/TimeChat-Captioner-GRPO-7B",
        note="AV caption model with timestamp behaviour.",
    ),
    "qwen_audio_vl": ModelSpec(
        name="qwen_audio_vl",
        kind="audio_video",
        requires_video=True,
        requires_audio=True,
        default_audio_model_path="Qwen/Qwen3-Omni-30B-A3B-Captioner",
        default_video_model_path="Qwen/Qwen3-VL-8B-Instruct",
        note="Qwen3-Omni-Captioner (audio-only, NO text prompt) + Qwen3-VL video.",
    ),
    "af3_vl": ModelSpec(
        name="af3_vl",
        kind="audio_video",
        requires_video=True,
        requires_audio=True,
        default_audio_model_path="nvidia/audio-flamingo-3",
        default_video_model_path="Qwen/Qwen3-VL-8B-Instruct",
        non_commercial=True,
        note="Audio Flamingo 3 audio caption + Qwen3-VL video. "
        "AF3 is NON-COMMERCIAL research only.",
    ),
    "secap_qwen": ModelSpec(
        name="secap_qwen",
        kind="audio_video",
        requires_video=True,
        requires_audio=True,
        default_audio_model_path="yaoxunxu/SECaps",
        default_video_model_path="Qwen/Qwen3-VL-8B-Instruct",
        note="SECap speech/audio-emotion caption (used directly) + Qwen3-VL "
        "video. Does NOT call Qwen3-Omni-Captioner.",
    ),
}


def supported_models() -> List[str]:
    return sorted(CAPTION_MODEL_SPECS)


def get_model_spec(caption_model: str) -> ModelSpec:
    try:
        return CAPTION_MODEL_SPECS[caption_model]
    except KeyError:
        raise ValueError(
            f"unknown --caption-model {caption_model!r}; "
            f"supported: {', '.join(supported_models())}"
        )


# ---------------------------------------------------------------------------
# Input boundary validation (pure — no model, no I/O)
# ---------------------------------------------------------------------------
def validate_inputs(
    caption_model: str,
    video: Optional[str],
    audio: Optional[str],
) -> ModelSpec:
    """Check the --video / --audio inputs a caption model requires.

    Raises ``ValueError`` with a clear message when a required input is missing.
    Returns the resolved ``ModelSpec`` so callers don't look it up twice.
    """
    spec = get_model_spec(caption_model)
    if spec.requires_video and not video:
        raise ValueError(
            f"caption model {caption_model!r} is a "
            f"{'audio+video' if spec.requires_audio else 'video/AV'} model and "
            f"requires --video."
        )
    if spec.requires_audio and not audio:
        raise ValueError(
            f"caption model {caption_model!r} requires --audio (audio+video "
            f"model)."
        )
    return spec


# ---------------------------------------------------------------------------
# Segment helper
# ---------------------------------------------------------------------------
def make_segment(
    segment_id: str = "s001",
    start: float = 0.0,
    end: float = 5.0,
    clip_path: Optional[str] = None,
    index: Optional[int] = None,
) -> Segment:
    """Build one ``Segment``. ``index`` defaults to the numeric part of s001 -> 1."""
    if index is None:
        digits = "".join(ch for ch in segment_id if ch.isdigit())
        index = int(digits) if digits else 1
    return Segment(
        segment_id=segment_id,
        index=index,
        start_time=round(float(start), 6),
        end_time=round(float(end), 6),
        clip_path=clip_path,
    )


# ---------------------------------------------------------------------------
# Normalization: raw caption (any shape) -> OmniCaption
# ---------------------------------------------------------------------------
def _raw_to_text(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, dict):
        return json.dumps(raw, ensure_ascii=False)
    return str(raw)


def _try_parse_json_object(raw: Any) -> Optional[dict]:
    """Return a dict if ``raw`` is (or contains) a JSON object, else ``None``.

    Accepts a dict directly, or a string that is JSON (tolerating ```json fences
    and leading/trailing prose by decoding from the first ``{``). Plain prose
    returns ``None``.
    """
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if not text:
        return None
    if "```" in text:
        fence = text.find("```")
        rest = text[fence + 3:]
        if "\n" in rest:
            rest = rest.split("\n", 1)[1]
        end = rest.rfind("```")
        if end != -1:
            rest = rest[:end]
        text = rest.strip()
    start = text.find("{")
    if start == -1:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _valid_enum(value: Any, allowed: tuple, default: str) -> str:
    return value if value in allowed else default


def _route_plain_text(data: Dict[str, Any], text: str, modality: str) -> None:
    """Place a plain-text caption into the right OmniCaption field(s) by modality.

    - ``av``    -> temporal_description (a neutral progression field describing
                   both streams; we never split what we can't split).
    - ``audio`` -> audio_description (audio evidence only; NEVER visual).
    - ``video`` -> temporal_description (visual/temporal; NEVER audio_description).
    """
    if modality == "audio":
        data["audio_description"] = text
    else:  # "av" or "video" -> temporal_description, no fabricated audio/visual
        data["temporal_description"] = text


def normalize_to_omni_caption(
    raw: Any,
    segment: Segment,
    video_id: str,
    *,
    source_caption_model: str,
    modality: str = "av",
    audio_source_model: Optional[str] = None,
    video_source_model: Optional[str] = None,
) -> OmniCaption:
    """Coerce any caption-model output into a valid ``OmniCaption``.

    ``raw`` may be a dict, a JSON string, plain text, or empty/malformed output.
    ``modality`` (``av`` / ``audio`` / ``video``) controls where plain text lands
    so an audio-only model never fabricates visual evidence and vice-versa. The
    trusted ``video_id`` / ``segment_id`` / ``time_range`` are always forced from
    ``segment``; a model-echoed ``segment_id`` / ``time_range`` is discarded.
    Extra provenance/debug fields are attached (``OmniCaption`` allows extras).
    """
    tr = [round(segment.start_time, 2), round(segment.end_time, 2)]
    raw_text = _raw_to_text(raw)
    preview = raw_text.strip()[:500]
    parsed = _try_parse_json_object(raw)

    data: Dict[str, Any] = {}
    status = "normalized"

    if parsed is not None:
        # Map only the recognized OmniCaption content fields, and only when they
        # have a usable type (a string field must be a str; the nested visual
        # fields must be dict/list). Anything else is ignored (never fabricated).
        vo = parsed.get("visual_objective")
        if isinstance(vo, dict):
            data["visual_objective"] = vo
        ve = parsed.get("visual_expression")
        if isinstance(ve, list):
            data["visual_expression"] = ve
        for key in ("audio_description", "temporal_description"):
            val = parsed.get(key)
            if isinstance(val, str) and val.strip():
                data[key] = val
        conf = parsed.get("confidence")
        if conf in _CONFIDENCE_VALUES:
            data["confidence"] = conf
        ev = parsed.get("evidence_strength")
        if ev in _EVIDENCE_VALUES:
            data["evidence_strength"] = ev
        if not data:
            # JSON decoded but carried no usable caption content -> salvage.
            status = "salvaged"
            _route_plain_text(data, preview or "(empty JSON caption)", modality)
    else:
        text = raw_text.strip()
        if not text:
            status = "salvaged"
            _route_plain_text(data, "(unparseable / empty caption)", modality)
        elif "{" in text or "[" in text:
            # Looked like JSON (a brace/bracket) but failed to decode -> malformed
            # output; salvage it as weak evidence rather than trust broken JSON.
            status = "salvaged"
            _route_plain_text(data, text, modality)
        else:
            # Genuine plain-text caption -> a valid (soft) observation.
            _route_plain_text(data, text, modality)

    # Trusted metadata — never from the model.
    data["segment_id"] = segment.segment_id
    data["time_range"] = tr
    data["video_id"] = video_id

    # Salvaged captions are pinned to soft confidence so the generator treats them
    # as weak evidence.
    if status == "salvaged":
        data["confidence"] = "low"
        data["evidence_strength"] = "weak"
    else:
        data.setdefault("confidence", "low")
        data.setdefault("evidence_strength", "ambiguous")
    data["confidence"] = _valid_enum(data.get("confidence"), _CONFIDENCE_VALUES, "low")
    data["evidence_strength"] = _valid_enum(
        data.get("evidence_strength"), _EVIDENCE_VALUES, "ambiguous"
    )

    # Provenance / debug extras (kept via OmniCaption's extra="allow").
    data["source_caption_model"] = source_caption_model
    data["caption_status"] = status
    data["raw_output_preview"] = preview
    if audio_source_model:
        data["audio_source_model"] = audio_source_model
    if video_source_model:
        data["video_source_model"] = video_source_model

    try:
        return OmniCaption.model_validate(data)
    except Exception:
        # Last resort: keep only the trusted text so a segment is never dropped.
        return OmniCaption(
            segment_id=segment.segment_id,
            video_id=video_id,
            time_range=tr,
            audio_description=data.get("audio_description", "")
            if isinstance(data.get("audio_description"), str) else "",
            temporal_description=data.get("temporal_description", "")
            if isinstance(data.get("temporal_description"), str) else preview,
            confidence="low",
            evidence_strength="weak",
            source_caption_model=source_caption_model,
            caption_status="salvaged",
            raw_output_preview=preview,
        )


def merge_audio_video_caption(
    audio_text: Any,
    video_text: Any,
    segment: Segment,
    video_id: str,
    *,
    audio_source_model: str,
    video_source_model: str,
    source_caption_model: str,
) -> OmniCaption:
    """Merge a separate audio caption + video caption into one ``OmniCaption``.

    The video model's text supplies the VISUAL evidence (normalized as a
    video-only caption -> visual/temporal fields); the audio model's text is put
    directly into ``audio_description`` as the AUDIO/speech-emotion evidence. This
    keeps the boundaries clean: the audio model never writes visual fields and the
    video model never writes ``audio_description``.
    """
    cap = normalize_to_omni_caption(
        video_text,
        segment,
        video_id,
        source_caption_model=source_caption_model,
        modality="video",
        audio_source_model=audio_source_model,
        video_source_model=video_source_model,
    )
    audio_str = _raw_to_text(audio_text).strip()
    if audio_str:
        cap.audio_description = audio_str
        cap.caption_status = "merged"
    return cap


# ---------------------------------------------------------------------------
# Downstream inputs + Gemini run (reuses the existing pipeline stages)
# ---------------------------------------------------------------------------
@dataclass
class DownstreamInputs:
    """The exact structures the existing Gemini downstream stages expect.

    ``generate_emotion_events(video_id, captions, client, segments)`` and
    ``generate_queries(video_id, captions, events, client, segments)`` both take a
    ``video_id``, a list of captions (``OmniCaption``) and a list of ``Segment``.
    """

    video_id: str
    segments: List[Segment]
    captions: List[OmniCaption]


def build_downstream_inputs(
    video_id: str,
    captions: List[OmniCaption],
    segments: List[Segment],
) -> DownstreamInputs:
    """Validate + package captions/segments for the Gemini downstream stages.

    Type-checks every caption/segment (so a wrong shape fails here, not deep
    inside Gemini calling code) and returns them in a struct that splats straight
    into ``generate_emotion_events`` / ``generate_queries``.
    """
    if not isinstance(video_id, str) or not video_id:
        raise ValueError("video_id must be a non-empty string")
    for c in captions:
        if not isinstance(c, OmniCaption):
            raise TypeError(f"caption is not an OmniCaption: {type(c).__name__}")
    for s in segments:
        if not isinstance(s, Segment):
            raise TypeError(f"segment is not a Segment: {type(s).__name__}")
    return DownstreamInputs(
        video_id=video_id, segments=list(segments), captions=list(captions)
    )


def run_downstream_gemini(
    inputs: DownstreamInputs,
    client,
    prompts_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run the REAL Gemini emotion-event + query-generation stages.

    ``client`` is a ``BaseLLMClient`` (a real ``GeminiLLMClient`` for the server
    integration run, or a fake in unit tests). Never mocks the stages. Returns the
    ``EmotionEventOutput``, the ``GenerationOutput`` and any warnings (e.g. zero
    queries, surfaced rather than hidden).
    """
    from .emotion_events import generate_emotion_events
    from .generation import generate_queries

    warnings: List[str] = []
    event_output = generate_emotion_events(
        inputs.video_id, inputs.captions, client, inputs.segments, prompts_dir
    )
    if not event_output.events:
        warnings.append(
            "emotion-event stage produced 0 events; no emotion to ground queries "
            "on (generation will return 0 queries)."
        )
    gen_output = generate_queries(
        inputs.video_id, inputs.captions, event_output.events, client,
        inputs.segments, prompts_dir,
    )
    if not gen_output.queries:
        warnings.append(
            "query-generation stage produced 0 queries for these captions."
        )
    return {"events": event_output, "generation": gen_output, "warnings": warnings}


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------
def save_outputs(
    output_dir: Path,
    *,
    raw_records: List[dict],
    captions: List[OmniCaption],
    events,
    generation,
    metadata: dict,
    final_queries: Optional[list] = None,
) -> Dict[str, str]:
    """Write every intermediate + final artefact under ``output_dir``.

    Returns a map of logical name -> written path (as str). Files:
    ``raw_caption_output.json``, ``normalized_captions.jsonl``,
    ``emotion_events.json``, ``generated_queries.json``, ``run_metadata.json`` and
    (optionally) ``final_queries.json``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, str] = {}

    def _dump_json(name: str, obj: Any) -> None:
        path = output_dir / name
        path.write_text(
            json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        written[name] = str(path)

    _dump_json("raw_caption_output.json", {"segments": raw_records})

    norm_path = output_dir / "normalized_captions.jsonl"
    write_jsonl(norm_path, captions)
    written["normalized_captions.jsonl"] = str(norm_path)

    _dump_json(
        "emotion_events.json",
        events.model_dump() if hasattr(events, "model_dump") else events,
    )
    _dump_json(
        "generated_queries.json",
        generation.model_dump() if hasattr(generation, "model_dump") else generation,
    )
    _dump_json("run_metadata.json", metadata)
    if final_queries is not None:
        _dump_json("final_queries.json", final_queries)
    return written


# ---------------------------------------------------------------------------
# Model runners (HEAVY — every import is lazy, inside the branch that needs it)
# ---------------------------------------------------------------------------
@dataclass
class CaptionModelOutput:
    """Raw output of one caption-model run for one segment.

    For ``av`` models ``raw_output`` holds the model text/dict and ``modality`` is
    ``"av"``. For ``audio_video`` models ``audio_text`` / ``video_text`` hold the
    two sub-model outputs to be merged.
    """

    modality: str = "av"
    raw_output: Any = None
    audio_text: Optional[str] = None
    video_text: Optional[str] = None
    source_caption_model: str = ""
    audio_source_model: Optional[str] = None
    video_source_model: Optional[str] = None
    debug: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunnerConfig:
    """Per-run knobs passed to every model runner."""

    caption_model_path: Optional[str] = None
    audio_model_path: Optional[str] = None
    video_model_path: Optional[str] = None
    device_map: str = "auto"
    attn_impl: Optional[str] = None
    max_new_tokens: int = 1024


def run_caption_model(
    caption_model: str,
    segment: Segment,
    *,
    video_path: Optional[str],
    audio_path: Optional[str],
    config: RunnerConfig,
    prompts_dir: Optional[Path] = None,
) -> CaptionModelOutput:
    """Dispatch to the per-model runner for one segment. Heavy deps stay lazy."""
    spec = get_model_spec(caption_model)
    if caption_model == "qwen3_omni":
        return _run_qwen3_omni(spec, segment, video_path, config, prompts_dir)
    if caption_model == "qwen_audio_vl":
        return _run_qwen_audio_vl(spec, segment, video_path, audio_path, config)
    if caption_model == "af3_vl":
        return _run_af3_vl(spec, segment, video_path, audio_path, config)
    if caption_model == "secap_qwen":
        return _run_secap_qwen(spec, segment, video_path, audio_path, config)
    if caption_model == "avocado":
        return _run_avocado(spec, segment, video_path, config)
    if caption_model == "timechat":
        return _run_timechat(spec, segment, video_path, config)
    raise ValueError(f"no runner for caption model {caption_model!r}")


def _run_qwen3_omni(
    spec: ModelSpec,
    segment: Segment,
    video_path: Optional[str],
    config: RunnerConfig,
    prompts_dir: Optional[Path],
) -> CaptionModelOutput:
    """Reuse the pipeline's own Qwen3-Omni captioner (video + text prompt)."""
    from .omni_captioning import (  # lazy: constructing is cheap/import-safe
        Qwen3OmniCaptioner,
        build_omni_caption_prompt,
    )

    model_path = config.caption_model_path or spec.default_model_path
    captioner = Qwen3OmniCaptioner(
        model_path=model_path,
        attn_implementation=config.attn_impl,
        device_map=config.device_map,
    )
    prompt = build_omni_caption_prompt([segment], prompts_dir)
    messages = Qwen3OmniCaptioner._build_messages_multi(prompt, [video_path or ""])
    raw = captioner.generate(messages)  # loads the model on first call (GPU)
    return CaptionModelOutput(
        modality="av", raw_output=raw, source_caption_model=model_path
    )


def _run_qwen3_vl_video(
    video_path: str, model_path: str, config: RunnerConfig
) -> str:
    """Caption a video clip's VISUALS with Qwen3-VL (video + text, no audio).

    Qwen3-VL is a vision-language model: it must receive video/image + text and
    must NOT receive audio. Returns plain caption text. Heavy imports are lazy.
    """
    import torch  # lazy
    from transformers import AutoModelForImageTextToText, AutoProcessor  # lazy
    from qwen_vl_utils import process_vision_info  # lazy

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype="auto",
        device_map=config.device_map,
        trust_remote_code=True,
        **({"attn_implementation": config.attn_impl} if config.attn_impl else {}),
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path},
                {"type": "text", "text": VIDEO_CAPTION_INSTRUCTION},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    images, videos = process_vision_info(messages)
    inputs = processor(
        text=[text], images=images, videos=videos, padding=True, return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=config.max_new_tokens)
    trimmed = gen[:, inputs["input_ids"].shape[1]:]
    out = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return (out[0] if out else "").strip()


def _run_qwen_omni_captioner_audio(audio_path: str, model_path: str) -> str:
    """Caption audio ONLY with Qwen3-Omni-Captioner — NO text prompt.

    Qwen3-Omni-Captioner is audio-only and does not accept a text instruction, so
    the user turn carries just the audio. Returns plain caption text.
    """
    import torch  # lazy
    from transformers import (  # lazy
        Qwen3OmniMoeForConditionalGeneration,
        Qwen3OmniMoeProcessor,
    )
    from qwen_omni_utils import process_mm_info  # lazy

    processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        model_path, dtype="auto", device_map="auto", trust_remote_code=True
    )
    # Audio-only turn: NO text part (Captioner takes no text prompt).
    conversations = [{"role": "user", "content": [{"type": "audio", "audio": audio_path}]}]
    text = processor.apply_chat_template(
        conversations, add_generation_prompt=True, tokenize=False
    )
    audios, images, videos = process_mm_info(conversations, use_audio_in_video=False)
    inputs = processor(
        text=text, audio=audios, images=images, videos=videos,
        return_tensors="pt", padding=True,
    ).to(model.device)
    with torch.no_grad():
        gen = model.generate(
            **inputs, return_audio=False, thinker_max_new_tokens=1024
        )
    text_ids = gen[0] if isinstance(gen, (tuple, list)) else gen
    seq = getattr(text_ids, "sequences", text_ids)
    out = processor.batch_decode(
        seq[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )
    return (out[0] if out else "").strip()


def _run_qwen_audio_vl(
    spec: ModelSpec,
    segment: Segment,
    video_path: Optional[str],
    audio_path: Optional[str],
    config: RunnerConfig,
) -> CaptionModelOutput:
    """Qwen3-Omni-Captioner (audio, no text) + Qwen3-VL (video) -> merge."""
    audio_model = config.audio_model_path or spec.default_audio_model_path
    video_model = config.video_model_path or spec.default_video_model_path
    video_text = _run_qwen3_vl_video(video_path or "", video_model, config)
    audio_text = _run_qwen_omni_captioner_audio(audio_path or "", audio_model)
    return CaptionModelOutput(
        modality="audio_video",
        audio_text=audio_text,
        video_text=video_text,
        source_caption_model=spec.name,
        audio_source_model=audio_model,
        video_source_model=video_model,
    )


def _run_af3_vl(
    spec: ModelSpec,
    segment: Segment,
    video_path: Optional[str],
    audio_path: Optional[str],
    config: RunnerConfig,
) -> CaptionModelOutput:
    """Audio Flamingo 3 (audio) + Qwen3-VL (video) -> merge.

    The Qwen3-VL video half is implemented; the AF3 audio half needs the NVIDIA
    ``audio-flamingo-3`` research repo (non-commercial only) whose inference API is
    not a plain ``transformers`` call, so it is left as a clear
    ``NotImplementedError`` rather than a guessed/broken call.
    """
    audio_model = config.audio_model_path or spec.default_audio_model_path
    video_model = config.video_model_path or spec.default_video_model_path
    # Run the well-defined half first so the failure message is specific.
    video_text = _run_qwen3_vl_video(video_path or "", video_model, config)
    _ = video_text  # computed but the merge can't complete without AF3 audio.
    raise NotImplementedError(
        "af3_vl audio half not wired: Audio Flamingo 3 "
        f"({audio_model}) is NON-COMMERCIAL research only and ships its own "
        "inference repo (not a plain transformers pipeline). Add an AF3 runner "
        "that loads the model per NVIDIA's audio-flamingo-3 instructions and "
        "returns an audio caption for --audio, then merge it with the Qwen3-VL "
        "video caption via merge_audio_video_caption(). See "
        "https://huggingface.co/nvidia/audio-flamingo-3"
    )


def _run_secap_qwen(
    spec: ModelSpec,
    segment: Segment,
    video_path: Optional[str],
    audio_path: Optional[str],
    config: RunnerConfig,
) -> CaptionModelOutput:
    """SECap (speech/audio-emotion, used directly) + Qwen3-VL (video) -> merge.

    Must NOT call Qwen3-Omni-Captioner. The Qwen3-VL video half is implemented;
    SECap needs its repo-specific checkpoint setup, so its runner is a clear
    ``NotImplementedError`` with instructions instead of a guessed API.
    """
    audio_model = config.audio_model_path or spec.default_audio_model_path
    video_model = config.video_model_path or spec.default_video_model_path
    video_text = _run_qwen3_vl_video(video_path or "", video_model, config)
    _ = video_text
    raise NotImplementedError(
        "secap_qwen audio half not wired: SECap "
        f"({audio_model}) needs repo-specific checkpoint setup (yaoxunxu/SECaps). "
        "Add a SECap runner that produces a speech/audio-emotion caption for "
        "--audio and use its output DIRECTLY as the audio evidence (do NOT call "
        "Qwen3-Omni-Captioner here), then merge with the Qwen3-VL video caption "
        "via merge_audio_video_caption(). See https://huggingface.co/yaoxunxu/SECaps"
    )


def _run_avocado(
    spec: ModelSpec,
    segment: Segment,
    video_path: Optional[str],
    config: RunnerConfig,
) -> CaptionModelOutput:
    """AVoCaDO AV captioner — needs the repo-specific runner."""
    model_path = config.caption_model_path or spec.default_model_path
    raise NotImplementedError(
        f"avocado runner not wired: AVoCaDO ({model_path}) needs its repo-specific "
        "inference code. Add a runner that captions --video and returns text/JSON, "
        "then normalize via normalize_to_omni_caption(..., modality='av'). See "
        "https://huggingface.co/AVoCaDO-Captioner/AVoCaDO"
    )


def _run_timechat(
    spec: ModelSpec,
    segment: Segment,
    video_path: Optional[str],
    config: RunnerConfig,
) -> CaptionModelOutput:
    """TimeChat AV captioner (timestamp behaviour) — needs the repo-specific runner."""
    model_path = config.caption_model_path or spec.default_model_path
    raise NotImplementedError(
        f"timechat runner not wired: TimeChat ({model_path}) needs its "
        "repo-specific inference code. Add a runner that captions --video and "
        "returns text/JSON, then normalize via "
        "normalize_to_omni_caption(..., modality='av'). See "
        "https://huggingface.co/yaolily/TimeChat-Captioner-GRPO-7B"
    )
