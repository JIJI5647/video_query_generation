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
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .io_utils import load_prompt_template, read_jsonl, write_jsonl
from .models import GenerationOutput, OmniCaption, Segment

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
_PROJECT_ROOT = Path(__file__).parent.parent

# Audio Flamingo 3 and SECap need dependency versions that conflict with the
# shared pipeline env (AF3: a newer transformers than this repo is pinned to;
# SECap: a 2023-era torch==2.0.0/transformers==4.29.0 legacy pin plus a
# repo-specific model tree), so each runs as a SUBPROCESS in its own venv
# rather than being imported in-process. See standalone_runners/af3_infer.py
# and third_party/SECap/scripts/standalone_inference.py. Overridable via env
# vars for a different server layout.
_AF3_ENV_PYTHON = os.environ.get(
    "AF3_ENV_PYTHON", str(_PROJECT_ROOT / "conda_envs" / "af3_env" / "bin" / "python")
)
_AF3_SCRIPT = str(_PROJECT_ROOT / "standalone_runners" / "af3_infer.py")
_SECAP_ENV_PYTHON = os.environ.get(
    "SECAP_ENV_PYTHON", str(_PROJECT_ROOT / "conda_envs" / "secap_env" / "bin" / "python")
)
_SECAP_REPO_DIR = Path(
    os.environ.get("SECAP_REPO_DIR", str(_PROJECT_ROOT / "third_party" / "SECap"))
)

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
        default_audio_model_path="nvidia/audio-flamingo-3-hf",
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
    default_confidence: str = "low",
) -> OmniCaption:
    """Coerce any caption-model output into a valid ``OmniCaption``.

    ``raw`` may be a dict, a JSON string, plain text, or empty/malformed output.
    ``modality`` (``av`` / ``audio`` / ``video``) controls where plain text lands
    so an audio-only model never fabricates visual evidence and vice-versa. The
    trusted ``video_id`` / ``segment_id`` / ``time_range`` are always forced from
    ``segment``; a model-echoed ``segment_id`` / ``time_range`` is discarded.
    Extra provenance/debug fields are attached (``OmniCaption`` allows extras).

    ``default_confidence`` sets the confidence for a genuinely-parsed plain-text
    caption that didn't declare its own (e.g. AVoCaDO's fused AV narrative, which
    is deliberately trusted higher than a naive free-text fallback since it's a
    model purpose-built/GRPO-optimized for cross-modal alignment). It never
    applies to the ``salvaged`` path (empty/malformed output stays "low"/"weak").
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
        data.setdefault("confidence", default_confidence)
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


# AVoCaDO's fused AV narrative and TimeChat's scene-timestamped narrative are both
# trusted higher than a naive free-text fallback (purpose-built cross-modal
# captioners, raw text passed through as-is in temporal_description) — see the
# caption_model -> confidence discussion in normalize_to_omni_caption's docstring.
DEFAULT_CONFIDENCE_BY_MODEL = {"avocado": "medium", "timechat": "medium"}


def normalize_caption_output(
    out: "CaptionModelOutput", segment: Segment, video_id: str, caption_model: str
) -> OmniCaption:
    """Coerce a caption runner/session output into an ``OmniCaption``.

    Dispatches on modality (``audio_video`` merges the two sub-model texts;
    everything else normalizes the single raw output) and applies the
    per-model default confidence. Shared by ``run_caption_generation_test.py``
    (per-segment) and ``run_caption_generation.py`` (batch).
    """
    if out.modality == "audio_video":
        return merge_audio_video_caption(
            out.audio_text, out.video_text, segment, video_id,
            audio_source_model=out.audio_source_model or "",
            video_source_model=out.video_source_model or "",
            source_caption_model=out.source_caption_model,
        )
    return normalize_to_omni_caption(
        out.raw_output, segment, video_id,
        source_caption_model=out.source_caption_model,
        modality=out.modality,
        audio_source_model=out.audio_source_model,
        video_source_model=out.video_source_model,
        default_confidence=DEFAULT_CONFIDENCE_BY_MODEL.get(caption_model, "low"),
    )


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
# Output writers / loaders — one pair per independent stage (caption generation
# / query generation / evaluation) so each stage can be run, cached and re-run
# on its own. Every stage also writes/passes through ``segments.jsonl`` so the
# next stage never needs to re-cut clips to know a segment's ``clip_path``.
# ---------------------------------------------------------------------------
def _dump_json(output_dir: Path, written: Dict[str, str], name: str, obj: Any) -> None:
    path = output_dir / name
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    written[name] = str(path)


def save_caption_outputs(
    output_dir: Path,
    *,
    raw_records: List[dict],
    captions: List[OmniCaption],
    segments: List[Segment],
    metadata: dict,
) -> Dict[str, str]:
    """Write stage-1 (caption generation) artefacts under ``output_dir``.

    Files: ``raw_caption_output.json``, ``normalized_captions.jsonl``,
    ``segments.jsonl``, ``run_metadata.json``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, str] = {}

    _dump_json(output_dir, written, "raw_caption_output.json", {"segments": raw_records})

    norm_path = output_dir / "normalized_captions.jsonl"
    write_jsonl(norm_path, captions)
    written["normalized_captions.jsonl"] = str(norm_path)

    seg_path = output_dir / "segments.jsonl"
    write_jsonl(seg_path, segments)
    written["segments.jsonl"] = str(seg_path)

    _dump_json(output_dir, written, "run_metadata.json", metadata)
    return written


def load_caption_outputs(captions_dir: Path) -> Tuple[List[Segment], List[OmniCaption]]:
    """Reload a stage-1 output dir's ``segments.jsonl`` + ``normalized_captions.jsonl``."""
    captions_dir = Path(captions_dir)
    segments = [Segment.model_validate(r) for r in read_jsonl(captions_dir / "segments.jsonl")]
    captions = [
        OmniCaption.model_validate(r)
        for r in read_jsonl(captions_dir / "normalized_captions.jsonl")
    ]
    return segments, captions


def save_generation_outputs(
    output_dir: Path,
    *,
    events,
    generation,
    segments: List[Segment],
    metadata: dict,
) -> Dict[str, str]:
    """Write stage-2 (query generation) artefacts under ``output_dir``.

    Files: ``emotion_events.json``, ``generated_queries.json``,
    ``segments.jsonl`` (passthrough, so stage 3 only needs ``--queries-dir``),
    ``generation_metadata.json``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, str] = {}

    _dump_json(
        output_dir, written, "emotion_events.json",
        events.model_dump() if hasattr(events, "model_dump") else events,
    )
    _dump_json(
        output_dir, written, "generated_queries.json",
        generation.model_dump() if hasattr(generation, "model_dump") else generation,
    )

    seg_path = output_dir / "segments.jsonl"
    write_jsonl(seg_path, segments)
    written["segments.jsonl"] = str(seg_path)

    _dump_json(output_dir, written, "generation_metadata.json", metadata)
    return written


def load_generation_outputs(queries_dir: Path) -> Tuple[List[Segment], GenerationOutput]:
    """Reload a stage-2 output dir's ``segments.jsonl`` + ``generated_queries.json``."""
    queries_dir = Path(queries_dir)
    segments = [Segment.model_validate(r) for r in read_jsonl(queries_dir / "segments.jsonl")]
    raw = json.loads((queries_dir / "generated_queries.json").read_text(encoding="utf-8"))
    generation = GenerationOutput.model_validate(raw)
    return segments, generation


def save_evaluation_outputs(
    output_dir: Path,
    *,
    final_queries: list,
    summary: dict,
    metadata: dict,
) -> Dict[str, str]:
    """Write stage-3 (evaluation/verification) artefacts under ``output_dir``.

    Files: ``final_queries.json``, ``verification_summary.json``,
    ``evaluation_metadata.json``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, str] = {}

    _dump_json(output_dir, written, "final_queries.json", final_queries)
    _dump_json(output_dir, written, "verification_summary.json", summary)
    _dump_json(output_dir, written, "evaluation_metadata.json", metadata)
    return written


def run_verification_stage(
    video_id: str, gen_output: GenerationOutput, segments: List[Segment], api_key: str,
    verification_model: Optional[str] = None, rewrite_model: Optional[str] = None,
) -> list:
    """Upload each query's grounded clip(s) and run the verify/rewrite loop on Gemini.

    Returns a list of ``QueryTrace`` dicts (one per query that was checked).
    """
    from .captioning import GeminiUploader
    from .llm_client import GeminiLLMClient
    from .workflow import run_query_pipeline

    client_kwargs: Dict[str, Any] = {"api_key": api_key}
    if verification_model:
        client_kwargs["verification_model"] = verification_model
    if rewrite_model:
        client_kwargs["rewrite_model"] = rewrite_model
    client = GeminiLLMClient(**client_kwargs)
    uploader = GeminiUploader(api_key=api_key)
    seg_by_id = {s.segment_id: s for s in segments}
    uploaded, segment_uris = [], {}
    try:
        for sid in sorted({sid for q in gen_output.queries for sid in q.segment_ids}):
            seg = seg_by_id.get(sid)
            if seg is None or not seg.clip_path:
                continue
            f = uploader.upload(seg.clip_path)
            uploaded.append(f)
            segment_uris[sid] = f.uri
        traces, _, _ = run_query_pipeline(video_id, gen_output, client, segment_uris)
    finally:
        for f in uploaded:
            uploader.delete(f)
    return [t.model_dump() for t in traces.values()]


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


def _load_qwen3_vl(model_path: str, config: RunnerConfig):
    """Load a Qwen3-VL model + processor once. Returns ``(model, processor)``."""
    from transformers import AutoModelForImageTextToText, AutoProcessor  # lazy

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype="auto",
        device_map=config.device_map,
        trust_remote_code=True,
        **({"attn_implementation": config.attn_impl} if config.attn_impl else {}),
    )
    return model, processor


def _qwen3_vl_generate(model, processor, video_path: str, config: RunnerConfig) -> str:
    """Caption one clip's VISUALS with an already-loaded Qwen3-VL model. Returns text."""
    import torch  # lazy
    from qwen_vl_utils import process_vision_info  # lazy

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


def _run_qwen3_vl_video(
    video_path: str, model_path: str, config: RunnerConfig
) -> str:
    """Load Qwen3-VL and caption ONE clip's visuals (per-call; reloads).

    Qwen3-VL is a vision-language model: it must receive video/image + text and
    must NOT receive audio. For many segments use
    ``batch_captioning.Qwen3VLVideoSession`` (loads once via ``_load_qwen3_vl``).
    """
    model, processor = _load_qwen3_vl(model_path, config)
    return _qwen3_vl_generate(model, processor, video_path, config)


def _run_subprocess_caption(
    cmd: List[str], cwd: Optional[str] = None, timeout: int = 900
) -> str:
    """Run a caption-model inference script in a SEPARATE Python env/process.

    Used for model families whose dependencies conflict with the shared pipeline
    env (Audio Flamingo 3 needs a newer ``transformers``; SECap needs a 2023
    legacy torch/transformers pin) — they run as a subprocess in their own venv,
    communicating the caption back over stdout between ``###CAPTION_START###`` /
    ``###CAPTION_END###`` markers, rather than being imported in-process.
    """
    result = subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout,
    )
    out = result.stdout
    start = out.find("###CAPTION_START###")
    end = out.find("###CAPTION_END###")
    if result.returncode != 0 or start == -1 or end == -1:
        raise RuntimeError(
            f"subprocess caption runner failed (exit {result.returncode}): "
            f"{' '.join(cmd)}\n"
            f"--- stdout (tail) ---\n{out[-2000:]}\n"
            f"--- stderr (tail) ---\n{result.stderr[-4000:]}"
        )
    return out[start + len("###CAPTION_START###"):end].strip()


def _load_qwen2_5_omni(model_path: str, config: RunnerConfig):
    """Load a Qwen2.5-Omni AV model + processor once (talker disabled).

    Returned ``(model, processor)`` is reusable across many segments — the batch
    captioner holds it for a whole run so the 7B weights load only once.
    """
    from transformers import (  # lazy
        Qwen2_5OmniForConditionalGeneration,
        Qwen2_5OmniProcessor,
    )

    load_kwargs: Dict[str, Any] = {
        "dtype": "auto",
        "device_map": config.device_map,
    }
    if config.attn_impl:
        load_kwargs["attn_implementation"] = config.attn_impl
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(model_path, **load_kwargs)
    if hasattr(model, "disable_talker"):
        model.disable_talker()
    processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
    return model, processor


def _qwen2_5_omni_generate(
    model, processor, video_path: str, prompt: str, config: RunnerConfig,
    system_prompt: Optional[str] = None,
) -> str:
    """Caption one clip with an already-loaded Qwen2.5-Omni AV model. Returns text."""
    import torch  # lazy
    from qwen_omni_utils import process_mm_info  # lazy

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
    messages.append({
        "role": "user",
        "content": [
            {"type": "video", "video": video_path},
            {"type": "text", "text": prompt},
        ],
    })
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(messages, use_audio_in_video=True)
    inputs = processor(
        text=text, audio=audios, images=images, videos=videos,
        return_tensors="pt", padding=True, use_audio_in_video=True,
    )
    inputs = inputs.to(model.device).to(model.dtype)
    with torch.no_grad():
        gen = model.generate(
            **inputs, use_audio_in_video=True, return_audio=False,
            thinker_max_new_tokens=config.max_new_tokens,
        )
    text_ids = gen[0] if isinstance(gen, (tuple, list)) else gen
    seq = getattr(text_ids, "sequences", text_ids)
    out = processor.batch_decode(
        seq[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )
    return (out[0] if out else "").strip()


def _run_qwen2_5_omni_av(
    video_path: str,
    model_path: str,
    prompt: str,
    config: RunnerConfig,
    system_prompt: Optional[str] = None,
) -> str:
    """Load a Qwen2.5-Omni AV captioner and caption ONE clip (per-call; reloads).

    Shared by AVoCaDO and TimeChat-Captioner-GRPO — both are fine-tunes of
    Qwen2.5-Omni-7B, called the same way per their reference inference scripts
    (talker disabled, ``process_mm_info`` with ``use_audio_in_video=True``); only
    the prompt (and optional system message) differs. For captioning many
    segments, use ``batch_captioning.Qwen25OmniAVSession`` instead, which loads
    the model once via ``_load_qwen2_5_omni`` and reuses it.
    """
    model, processor = _load_qwen2_5_omni(model_path, config)
    return _qwen2_5_omni_generate(
        model, processor, video_path, prompt, config, system_prompt=system_prompt
    )


def _load_qwen_omni_captioner(model_path: str):
    """Load Qwen3-Omni-Captioner (audio-only) model + processor once."""
    from transformers import (  # lazy
        Qwen3OmniMoeForConditionalGeneration,
        Qwen3OmniMoeProcessor,
    )

    processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        model_path, dtype="auto", device_map="auto", trust_remote_code=True
    )
    return model, processor


def _qwen_omni_captioner_generate(model, processor, audio_path: str) -> str:
    """Caption one audio clip with an already-loaded Qwen3-Omni-Captioner (no text prompt)."""
    import torch  # lazy
    from qwen_omni_utils import process_mm_info  # lazy

    # Audio-only turn: NO text part (Captioner takes no text prompt).
    conversations = [{"role": "user", "content": [{"type": "audio", "audio": audio_path}]}]
    text = processor.apply_chat_template(
        conversations, add_generation_prompt=True, tokenize=False
    )
    audios, images, videos = process_mm_info(conversations, use_audio_in_video=False)
    inputs = processor(
        text=text, audio=audios, images=images, videos=videos,
        return_tensors="pt", padding=True,
    ).to(model.device).to(model.dtype)
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


def _run_qwen_omni_captioner_audio(audio_path: str, model_path: str) -> str:
    """Load Qwen3-Omni-Captioner and caption ONE audio clip (per-call; reloads).

    Qwen3-Omni-Captioner is audio-only and does not accept a text instruction, so
    the user turn carries just the audio. For many segments use
    ``batch_captioning.QwenOmniCaptionerAudioSession`` (loads once).
    """
    model, processor = _load_qwen_omni_captioner(model_path)
    return _qwen_omni_captioner_generate(model, processor, audio_path)


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

    AF3's ``transformers`` HF integration needs a newer version than this repo's
    shared env is pinned to, so its audio half runs as a subprocess in
    ``conda_envs/af3_env`` (see ``standalone_runners/af3_infer.py``).
    NON-COMMERCIAL research use only (NVIDIA audio-flamingo-3 license).
    """
    audio_model = config.audio_model_path or spec.default_audio_model_path
    video_model = config.video_model_path or spec.default_video_model_path
    video_text = _run_qwen3_vl_video(video_path or "", video_model, config)
    audio_text = _run_subprocess_caption(
        [_AF3_ENV_PYTHON, _AF3_SCRIPT, audio_path or "", audio_model]
    )
    return CaptionModelOutput(
        modality="audio_video", audio_text=audio_text, video_text=video_text,
        source_caption_model=spec.name, audio_source_model=audio_model,
        video_source_model=video_model,
    )


def _run_secap_qwen(
    spec: ModelSpec,
    segment: Segment,
    video_path: Optional[str],
    audio_path: Optional[str],
    config: RunnerConfig,
) -> CaptionModelOutput:
    """SECap (speech/audio-emotion, used directly) + Qwen3-VL (video) -> merge.

    SECap needs a 2023 legacy torch==2.0.0/transformers==4.29.0 pin plus its own
    repo-specific model tree (``third_party/SECap``), so it runs as a subprocess
    in ``conda_envs/secap_env`` (see
    ``third_party/SECap/scripts/standalone_inference.py``). Never calls
    Qwen3-Omni-Captioner — SECap's own output IS the audio evidence.

    Note: SECap is a MANDARIN speech-emotion captioner (chinese-hubert +
    chinese-llama-7b, Chinese prompt/output); on non-Chinese audio (e.g. this
    pipeline's English pilot-study clips) its captions are expected to be
    unreliable — a language-domain mismatch, not a wiring bug.
    """
    audio_model = config.audio_model_path or spec.default_audio_model_path
    video_model = config.video_model_path or spec.default_video_model_path
    video_text = _run_qwen3_vl_video(video_path or "", video_model, config)
    audio_text = _run_subprocess_caption(
        [
            _SECAP_ENV_PYTHON, "standalone_inference.py",
            "--wavdir", audio_path or "",
        ],
        cwd=str(_SECAP_REPO_DIR / "scripts"),
    )
    return CaptionModelOutput(
        modality="audio_video", audio_text=audio_text, video_text=video_text,
        source_caption_model=spec.name, audio_source_model=audio_model,
        video_source_model=video_model,
    )


# One of AVoCaDO's own 7 paraphrased instructions (its inference.py picks one at
# random; pinned here for reproducibility). All 7 are semantically identical.
_AVOCADO_PROMPT = (
    "Provide a comprehensive description of all the content in the video, "
    "leaving out no details. Be sure to include as much of the audio "
    "information as possible, and ensure that your descriptions of the audio "
    "and video are closely aligned."
)
_AVOCADO_SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating "
    "text and speech."
)


def _run_avocado(
    spec: ModelSpec,
    segment: Segment,
    video_path: Optional[str],
    config: RunnerConfig,
) -> CaptionModelOutput:
    """AVoCaDO AV captioner (Qwen2.5-Omni-7B fine-tune; github.com/AVoCaDO-Captioner/AVoCaDO)."""
    model_path = config.caption_model_path or spec.default_model_path
    raw = _run_qwen2_5_omni_av(
        video_path or "", model_path, _AVOCADO_PROMPT, config,
        system_prompt=_AVOCADO_SYSTEM_PROMPT,
    )
    return CaptionModelOutput(modality="av", raw_output=raw, source_caption_model=model_path)


# TimeChat-Captioner-GRPO-7B's own reference prompt (its HF model card example).
_TIMECHAT_PROMPT = (
    "Thoroughly describe everything in the video, capturing every detail. "
    "Include as much information from the audio as possible, and ensure that "
    "the descriptions of both audio and video are well-coordinated."
)


def _try_parse_json_array(raw: str) -> Optional[list]:
    """Return a list if ``raw`` is (or contains) a top-level JSON array, else ``None``.

    Only treats ``raw`` as array-shaped when a ``[`` occurs before any ``{`` (a
    top-level object containing an incidental list value doesn't count). Tolerates
    ```json fences and leading/trailing prose, same as ``_try_parse_json_object``.
    """
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
    bracket = text.find("[")
    brace = text.find("{")
    if bracket == -1 or (brace != -1 and brace < bracket):
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[bracket:])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, list) else None


def _fold_timechat_scenes(scenes: list) -> Optional[str]:
    """Fold TimeChat's per-timestamp scene dicts into one plain-text description.

    Keeps ``segment_detail_caption`` (core visual/action content), ``storyline``
    (the model's own emotion/narrative reading) and ``acoustics_content`` (audio
    evidence — legitimate here since TimeChat is a combined AV model). Drops
    ``camera_state``/``shooting_style`` (cinematography, not useful signal) and
    ``video_background`` (redundant with the detail caption). Also drops
    ``speech_content`` — the real pipeline already has its own dedicated
    transcript source, so a second, model-specific transcription is unwanted
    here. Returns ``None`` if no scene has usable content (caller falls back to
    the existing salvage behaviour).
    """
    lines = []
    for scene in scenes:
        if not isinstance(scene, dict):
            continue
        detail = scene.get("segment_detail_caption")
        if not isinstance(detail, str) or not detail.strip():
            continue
        # No "[" / "{" in the folded text: normalize_to_omni_caption's salvage
        # heuristic treats either as a sign of failed-to-parse JSON, so a
        # bracketed timestamp prefix would get this genuinely-parsed content
        # salvaged right back into the bug this fold exists to avoid.
        timestamp = scene.get("timestamp")
        prefix = f"({timestamp}) " if isinstance(timestamp, str) and timestamp.strip() else ""
        line = f"{prefix}{detail.strip()}"
        storyline = scene.get("storyline")
        if isinstance(storyline, str) and storyline.strip():
            line += f" (Storyline: {storyline.strip()})"
        acoustics = scene.get("acoustics_content")
        if isinstance(acoustics, str) and acoustics.strip():
            line += f" (Audio: {acoustics.strip()})"
        lines.append(line)
    return "\n".join(lines) if lines else None


def _run_timechat(
    spec: ModelSpec,
    segment: Segment,
    video_path: Optional[str],
    config: RunnerConfig,
) -> CaptionModelOutput:
    """TimeChat-Captioner-GRPO-7B AV captioner (timestamp behaviour; Qwen2.5-Omni-7B fine-tune).

    The model card notes it targets clips up to ~1 minute for its time-aware,
    multi-scene captions; our 5s pipeline segments are well within that range.
    TimeChat's own selling point is returning a JSON ARRAY of per-timestamp scene
    dicts rather than one flat caption, so its raw output is folded into one
    plain-text string (``_fold_timechat_scenes``) before being handed to the
    generic normalizer — otherwise ``normalize_to_omni_caption`` would only see
    the first scene and truncate the rest (see docs/progress_log.md).
    """
    model_path = config.caption_model_path or spec.default_model_path
    raw = _run_qwen2_5_omni_av(video_path or "", model_path, _TIMECHAT_PROMPT, config)
    scenes = _try_parse_json_array(raw)
    if scenes is not None:
        folded = _fold_timechat_scenes(scenes)
        if folded is not None:
            raw = folded
    return CaptionModelOutput(modality="av", raw_output=raw, source_caption_model=model_path)
