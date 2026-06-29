"""Qwen3-Omni structured captioning backend — one segment per prompt.

Design / safety constraints (this machine cannot run Qwen3-Omni):

- LAZY EVERYTHING. ``torch``, ``transformers`` and ``qwen_omni_utils`` are
  imported, and the 30B model is loaded, only inside ``Qwen3OmniCaptioner._
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

Captions are OBSERVATION-ONLY (no emotion). Emotion is judged later in the Gemini
emotion-event stage. ``OmniCaption`` is consumed directly downstream.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

from .io_utils import load_prompt_template
from .models import (
    OMNI_REQUIRED_FIELDS,
    OmniCaption,
    Segment,
)

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

DEFAULT_MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

# Sampling defaults (spec §4.5). Plain dict consumed lazily inside the captioner
# (mapped to the transformers ``thinker_*`` generate kwargs).
DEFAULT_SAMPLING_PARAMS: Dict[str, Any] = {
    "temperature": 0.0,
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
    """Anything that turns pre-built chat messages into raw model text.

    ``generate_many`` runs N conversations in one batched call (order preserved);
    ``generate`` is the single-conversation convenience.
    """

    def generate(self, messages: list) -> str:  # pragma: no cover
        ...

    def generate_many(self, messages_list: List[list]) -> List[str]:  # pragma: no cover
        ...


# ---------------------------------------------------------------------------
# Prompt construction (N segments per prompt)
# ---------------------------------------------------------------------------
def build_omni_caption_prompt(
    segments: List[Segment], prompts_dir: Optional[Path] = None
) -> str:
    """Build the structured caption prompt for a chunk of N segments' clips.

    The clips are provided to the model in the same order as ``segments``; the
    prompt enumerates the Clip -> segment_id / time_range mapping and asks for a
    JSON array of one caption per clip. Works for N == 1 (array of one).
    """
    template = load_prompt_template(
        prompts_dir or _PROMPTS_DIR, "omni_caption_prompt.txt"
    )
    lines = [
        f"Clip {i} -> {s.segment_id} ({s.start_time:.2f}-{s.end_time:.2f}s)"
        for i, s in enumerate(segments, 1)
    ]
    return template.replace("{segment_list}", "\n".join(lines))


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


def extract_caption_list(raw_text: str) -> list:
    """Pull a JSON array of caption objects out of a raw model response.

    Tolerates markdown fences and surrounding prose. A bare single object is
    wrapped into a one-element list (some models drop the array for N == 1).
    Raises ``CaptionParseError(json_parse_error)`` if nothing decodes.
    """
    if not raw_text or not raw_text.strip():
        raise CaptionParseError("json_parse_error", "empty response", raw_text or "")
    text = raw_text.strip()
    if "```" in text:
        fence = text.find("```")
        rest = text[fence + 3 :]
        if "\n" in rest:
            rest = rest.split("\n", 1)[1]
        end = rest.rfind("```")
        if end != -1:
            rest = rest[:end]
        text = rest.strip()

    # Decode whichever JSON value (array or object) appears first.
    starts = [p for p in (text.find("["), text.find("{")) if p != -1]
    if not starts:
        raise CaptionParseError(
            "json_parse_error", "no JSON array/object in output", raw_text
        )
    try:
        value, _ = json.JSONDecoder().raw_decode(text[min(starts):])
    except json.JSONDecodeError as e:
        raise CaptionParseError("json_parse_error", str(e), raw_text) from e
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return value
    raise CaptionParseError(
        "json_parse_error", "top-level JSON is neither array nor object", raw_text
    )


def _match_items_to_segments(
    items: list, segments: List[Segment]
) -> Dict[str, dict]:
    """Map decoded caption objects to segments (echoed segment_id, else position).

    Returns ``segment_id -> raw item dict`` for every segment the model output
    can be tied to. Used by the salvage path so a segment whose object failed
    strict validation can still be turned into a best-effort caption.
    """
    by_id = {s.segment_id: s for s in segments}
    matched: Dict[str, dict] = {}
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        seg = by_id.get(item.get("segment_id"))
        if seg is None and idx < len(segments):
            seg = segments[idx]  # fall back to clip order
        if seg is None or seg.segment_id in matched:
            continue
        matched[seg.segment_id] = item
    return matched


def salvage_caption(
    item: Optional[dict], raw_text: str, segment: Segment, video_id: str
) -> OmniCaption:
    """Best-effort OmniCaption for a segment whose output failed strict parsing.

    The model always produces *some* format errors; rather than drop the segment
    (so generation never hears about that moment), we keep whatever decoded —
    ``OmniCaption`` allows extra keys and defaults the rest — force the trusted
    metadata, and pin ``confidence=low`` / ``evidence_strength=weak`` so the
    generator treats it as soft evidence. When nothing decoded, the raw model
    text is stuffed into ``temporal_description`` so the segment is at least
    described. A ``caption_status="salvaged"`` marker is carried for debug/export.
    """
    tr = [round(segment.start_time, 2), round(segment.end_time, 2)]
    data: Dict[str, Any] = dict(item) if isinstance(item, dict) else {}
    data["segment_id"] = segment.segment_id
    data["time_range"] = tr
    data["video_id"] = video_id
    data["caption_status"] = "salvaged"
    # Force the two enum fields to a valid soft value (a bad echo must not fail
    # validation — that is the whole point of salvaging).
    if data.get("confidence") not in ("high", "medium", "low"):
        data["confidence"] = "low"
    data["evidence_strength"] = "weak"
    fallback = (raw_text or "").strip()[:500] or "(unparseable)"
    if not (str(data.get("audio_description") or "")).strip() and not (
        str(data.get("temporal_description") or "")
    ).strip():
        data["temporal_description"] = fallback
    try:
        return OmniCaption.model_validate(data)
    except Exception:
        # Last resort: a minimal valid caption keeping only trusted text so the
        # segment is NEVER dropped, whatever odd shape the nested fields had.
        audio = data.get("audio_description")
        return OmniCaption(
            segment_id=segment.segment_id,
            video_id=video_id,
            time_range=tr,
            audio_description=audio if isinstance(audio, str) else "",
            temporal_description=fallback,
            confidence="low",
            evidence_strength="weak",
            caption_status="salvaged",
        )


def parse_captions(
    raw_text: str, segments: List[Segment], video_id: str
) -> Dict[str, OmniCaption]:
    """Parse N captions from one model output, keyed by segment_id.

    Each object is matched to a segment by its echoed ``segment_id`` (falling
    back to clip position when the echo is missing/unknown), then has its
    metadata forced from the trusted segment and is validated. Only segments
    with a valid caption appear in the result; missing/invalid ones are left to
    the caller (raw dump + retry next run). Raises ``CaptionParseError`` only if
    the whole array fails to decode.
    """
    items = extract_caption_list(raw_text)
    by_id = {s.segment_id: s for s in segments}
    out: Dict[str, OmniCaption] = {}
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        seg = by_id.get(item.get("segment_id"))
        if seg is None and idx < len(segments):
            seg = segments[idx]  # fall back to clip order
        if seg is None or seg.segment_id in out:
            continue
        if missing_required_fields(item):
            continue
        item["segment_id"] = seg.segment_id
        item["time_range"] = [round(seg.start_time, 2), round(seg.end_time, 2)]
        item["video_id"] = video_id
        try:
            out[seg.segment_id] = OmniCaption.model_validate(item)
        except Exception:
            continue
    return out


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
# Per-segment + per-video captioning (with resume)
# ---------------------------------------------------------------------------
def _chunks(items: list, size: int) -> List[list]:
    return [items[i : i + size] for i in range(0, len(items), max(1, size))]


def _resolve_cache(
    video_id: str,
    segments: List[Segment],
    cache_dir: Path,
    resume: bool,
    overwrite: bool,
):
    """Split segments into (cached results, segments needing generation).

    Logs skip / regenerate decisions. Segments without a clip are dropped.
    """
    cached: Dict[str, OmniCaption] = {}
    to_generate: List[Segment] = []
    for segment in segments:
        if not segment.clip_path:
            continue
        if resume and not overwrite:
            cap, reason = read_valid_cache(
                caption_cache_path(cache_dir, video_id, segment.segment_id)
            )
            if cap is not None:
                print(f"[caption] skip existing: video_id={video_id} "
                      f"segment_id={segment.segment_id}")
                cached[segment.segment_id] = cap
                continue
            if reason != "not_found":
                print(f"[caption] regenerate invalid cache: video_id={video_id} "
                      f"segment_id={segment.segment_id} reason={reason}")
        to_generate.append(segment)
    return cached, to_generate


def _write_chunk_captions(
    video_id: str,
    chunk: List[Segment],
    raw_text: str,
    cache_dir: Path,
    raw_dir: Path,
    results_by_id: Dict[str, OmniCaption],
) -> None:
    """Parse N captions from one prompt's output; cache valid ones, salvage the rest.

    A segment with a fully valid caption is cached and reused on rerun. A segment
    the model skipped or returned in a bad format is NOT cached (so a later run
    can still regenerate a clean caption), but a best-effort ``salvage_caption``
    is still added to the results so it reaches generation — a format error never
    makes a moment invisible to the generator, and never aborts the chunk. The
    raw model text is dumped either way for debugging.
    """
    try:
        parsed = parse_captions(raw_text, chunk, video_id)
    except CaptionParseError:
        parsed = {}  # whole array undecodable -> every segment is salvaged below
    try:
        items = extract_caption_list(raw_text)
    except CaptionParseError:
        items = []
    matched = _match_items_to_segments(items, chunk)
    for segment in chunk:
        caption = parsed.get(segment.segment_id)
        if caption is not None:
            atomic_write_json(
                caption_cache_path(cache_dir, video_id, segment.segment_id),
                caption.model_dump(),
            )
            results_by_id[segment.segment_id] = caption
            continue
        # Salvage: keep the moment for generation, raw-dump, leave uncached.
        raw_path = save_raw_output(
            raw_dir, video_id, segment.segment_id, raw_text,
            "salvaged_for_generation",
        )
        results_by_id[segment.segment_id] = salvage_caption(
            matched.get(segment.segment_id), raw_text, segment, video_id
        )
        print(f"[caption] salvaged (fed to generation, not cached): "
              f"video_id={video_id} segment_id={segment.segment_id} "
              f"raw_saved={raw_path}")


def caption_video_omni(
    video_id: str,
    segments: List[Segment],
    captioner: CaptionerProtocol,
    cache_dir: Path,
    raw_dir: Path,
    resume: bool = True,
    overwrite: bool = False,
    caption_batch_size: int = DEFAULT_CAPTION_BATCH_SIZE,
    caption_parallel: int = 1,
    prompts_dir: Optional[Path] = None,
) -> List[OmniCaption]:
    """Caption every segment of one video, with two orthogonal batching dims.

    - ``caption_batch_size`` = segments packed into ONE prompt (the model sees N
      segment clips at once and returns N captions, each mapped back to its
      segment_id). N == 1 is one segment per prompt.
    - ``caption_parallel`` = how many such prompts run together in ONE model
      ``generate`` call (throughput). 1 = one prompt per call.

    Resume is applied first (cached segments skipped, no model call); only
    cache-miss segments are grouped. A segment the model skips/garbles is raw
    dumped, salvaged into a best-effort low-confidence caption (so it still
    reaches generation), and left uncached so a later run can regenerate a clean
    one — it never aborts the video. Only a whole-group ``generate`` error leaves
    a segment with no caption at all (raw dumped, retried next run).
    """
    results_by_id: Dict[str, OmniCaption] = {}
    cached, to_generate = _resolve_cache(
        video_id, segments, cache_dir, resume, overwrite
    )
    results_by_id.update(cached)

    seg_per_prompt = max(1, caption_batch_size)
    prompts_per_call = max(1, caption_parallel)
    prompt_chunks = _chunks(to_generate, seg_per_prompt)  # each chunk = 1 prompt

    for group in _chunks(prompt_chunks, prompts_per_call):  # prompts per generate
        for chunk in group:
            ids = ", ".join(s.segment_id for s in chunk)
            print(f"[caption] generate: video_id={video_id} segment_ids=[{ids}] "
                  f"(segments/prompt={len(chunk)})")
        messages_list = [
            Qwen3OmniCaptioner._build_messages_multi(
                build_omni_caption_prompt(chunk, prompts_dir),
                [s.clip_path or "" for s in chunk],
            )
            for chunk in group
        ]
        try:
            raw_texts = captioner.generate_many(messages_list)
        except Exception as e:  # whole group failed -> raw dump, skip (retry next run)
            print(f"[caption] generate failed for {len(group)} prompt(s): {e}")
            for chunk in group:
                for segment in chunk:
                    save_raw_output(
                        raw_dir, video_id, segment.segment_id, "",
                        f"generate_error: {e}",
                    )
            continue

        for chunk, raw_text in zip(group, raw_texts):
            _write_chunk_captions(
                video_id, chunk, raw_text, cache_dir, raw_dir, results_by_id
            )

    # Reassemble in original segment order.
    return [
        results_by_id[s.segment_id]
        for s in segments
        if s.segment_id in results_by_id
    ]


# ---------------------------------------------------------------------------
# The actual Qwen3-Omni backend (lazy model load — never at import time)
# ---------------------------------------------------------------------------
@dataclass
class Qwen3OmniCaptioner:
    """Qwen3-Omni captioner on the HuggingFace transformers engine. Loads on first use.

    Constructing this object is cheap and import-safe: it stores config only. No
    ``torch`` / ``transformers`` / ``qwen_omni_utils`` import happens until
    ``_ensure_model`` runs (inside the first ``caption`` call), so nothing here
    touches a GPU or downloads weights on this machine.

    Inference is a pure HuggingFace ``Qwen3OmniMoeForConditionalGeneration``
    (``device_map="auto"`` + the talker disabled for text-only captioning). The
    public ``caption`` / ``generate`` contract is unchanged.
    """

    model_path: str = DEFAULT_MODEL_PATH
    sampling_params: Dict[str, Any] = field(
        default_factory=lambda: dict(DEFAULT_SAMPLING_PARAMS)
    )
    use_audio_in_video: bool = True
    # Force the qwen_omni_utils video reader so it never tries torchcodec
    # (which often fails to load on mismatched CUDA/ffmpeg). "" leaves the
    # library's auto-detection alone. Set via FORCE_QWENVL_VIDEO_READER.
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
        # Pin the qwen_omni_utils video reader BEFORE any video is decoded, so it
        # never falls through to torchcodec. Set before process_mm_info import.
        if self.video_reader_backend:
            os.environ["FORCE_QWENVL_VIDEO_READER"] = self.video_reader_backend
        import time as _time
        t0 = _time.perf_counter()
        print(f"[model] loading {self.model_path} (transformers)...")
        self._ensure_model_transformers()
        print(f"[model] loaded in {_time.perf_counter() - t0:.1f}s "
              f"(one-time; reused for all videos/segments)")

    def _ensure_model_transformers(self) -> None:
        # Lazy imports — keep these OFF the module import path.
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
        # Captioning needs TEXT only — drop the audio-generating talker to save
        # memory/time if the build supports it.
        if hasattr(self._model, "disable_talker"):
            try:
                self._model.disable_talker()
            except Exception:
                pass
        self._processor = Qwen3OmniMoeProcessor.from_pretrained(self.model_path)

    @staticmethod
    def _build_messages_multi(prompt_text: str, clip_paths: List[str]) -> list:
        """One user turn: N segment videos (audio included) then the text prompt.

        Used for captioning (one clip) and for verification/rewrite (a query's
        grounded segment clip(s)). Empty ``clip_paths`` is a text-only turn.
        """
        content = [{"type": "video", "video": p} for p in clip_paths if p]
        content.append({"type": "text", "text": prompt_text})
        return [{"role": "user", "content": content}]

    # -- inference -----------------------------------------------------------
    def generate(self, messages: list) -> str:
        """Run inference on one pre-built conversation; return raw model text.

        Shared by captioning and the ``QwenOmniLLMClient`` (verify/rewrite).
        """
        out = self.generate_many([messages])
        return out[0] if out else ""

    def generate_many(self, messages_list: List[list]) -> List[str]:
        """Run inference on N pre-built conversations in ONE batched call.

        Output order matches input order. Each conversation is independent (its
        own clip(s) + prompt); used both for multi-segment caption prompts and to
        run several prompts in parallel per ``generate``.
        """
        if not messages_list:
            return []
        self._ensure_model()
        return self._caption_transformers_batch(messages_list)

    # -- transformers helpers (shared by the single + batch paths) -----------
    def _prep_transformers_inputs(self, conversations):
        """Build model inputs from one conversation or a list of conversations.

        ``apply_chat_template`` / ``process_mm_info`` accept either form; a list
        yields a padded batch. Mirrors the official Qwen3-Omni example.
        """
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
        """Generate text for a (possibly batched) ``inputs`` and decode per row.

        generate returns (text_ids, audio); with thinker_return_dict_in_generate
        the ids are under ``.sequences``. We ignore the audio (text-only caption)
        and slice off the prompt before decoding. Returns one string per row.
        """
        import torch

        sp = self.sampling_params
        # Confirmed-safe kwargs from the official Qwen3-Omni examples (text-only,
        # talker disabled, batch-safe).
        base_kwargs: Dict[str, Any] = {
            "return_audio": False,
            "thinker_return_dict_in_generate": True,
            "use_audio_in_video": self.use_audio_in_video,
        }
        # Sampling control via the Omni thinker_* knobs. Some builds may not
        # accept these; if so we fall back to the model's generation defaults
        # rather than failing the caption.
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

        # Return is (text_ids, audio); talker-disabled builds may return only
        # text_ids. text_ids is a ModelOutput (.sequences) or a plain tensor.
        text_ids = gen_out[0] if isinstance(gen_out, (tuple, list)) else gen_out
        seq = getattr(text_ids, "sequences", text_ids)
        return self._processor.batch_decode(
            seq[:, prompt_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    def _caption_transformers_batch(self, messages_list: List[list]) -> List[str]:
        inputs = self._prep_transformers_inputs(messages_list)
        return self._run_transformers_generate(inputs)


# ---------------------------------------------------------------------------
# LLM client for verify / rewrite on Qwen3-Omni (duck-typed to BaseLLMClient)
# ---------------------------------------------------------------------------
class QwenOmniLLMClient:
    """A ``BaseLLMClient``-compatible client backed by a Qwen3-Omni engine.

    Implements ``generate_json(prompt, schema_name, video_uri)`` so the existing
    verify / rewrite code can run on Qwen3-Omni instead of Gemini. Here
    ``video_uri`` is LOCAL clip path(s) — no Files API upload: ``None`` (text
    only), a single path, or a list of paths. The model watches those clips and
    returns parsed JSON; markdown fences / surrounding prose are tolerated.

    Shares the already-loaded model with the captioner (same ``Qwen3OmniCaptioner``
    instance) so the weights load once. Deliberately NOT a subclass of
    ``BaseLLMClient`` — that keeps this module free of the google-genai import
    (duck typing is enough; the verify/rewrite code never isinstance-checks).
    """

    def __init__(self, engine: "Qwen3OmniCaptioner", max_retries: int = 2) -> None:
        self._engine = engine
        self.max_retries = max(1, max_retries)

    def generate_json(
        self, prompt: str, schema_name: str, video_uri=None
    ) -> Dict[str, Any]:
        if video_uri is None:
            clip_paths: List[str] = []
        elif isinstance(video_uri, str):
            clip_paths = [video_uri]
        else:
            clip_paths = list(video_uri)
        messages = Qwen3OmniCaptioner._build_messages_multi(prompt, clip_paths)

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

    @staticmethod
    def _clip_paths(video_uri) -> List[str]:
        if video_uri is None:
            return []
        if isinstance(video_uri, str):
            return [video_uri]
        return list(video_uri)

    def generate_json_many(
        self, prompts: List[str], schema_name: str, video_uris=None
    ) -> List[Dict[str, Any]]:
        """Run N verify/rewrite prompts in ONE batched model forward.

        Each prompt keeps its own clip(s) (``video_uris[i]``); the shared
        Qwen3-Omni engine runs them as one ``generate_many`` call (the same
        batching caption uses). An item whose output won't parse falls back to a
        single retried ``generate_json`` so one bad query never fails the batch.
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
                # Retry just this item through the single-call retry path.
                out.append(self.generate_json(prompt, schema_name, video_uri=uri))
        return out

    def usage_report(self) -> Dict[str, Any]:
        """Interface parity with GeminiLLMClient (local model: no token counts)."""
        empty = {
            "calls": 0, "prompt_tokens": 0,
            "candidates_tokens": 0, "total_tokens": 0,
        }
        return {"by_stage": {}, "total": dict(empty)}
