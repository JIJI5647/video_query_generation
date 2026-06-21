"""WhisperX dialogue transcription (B3).

Transcribes a whole video into sentence-level lines with precise (word-aligned)
timestamps. The result is spliced into the generation prompt as extra context;
it never enters the caption stage. Non-verbal sounds (shouting, crying, etc.)
are NOT produced here — those stay in the caption ``sound`` field.

WhisperX (and torch) are heavy optional dependencies; the import is isolated so
the rest of the pipeline runs even when transcription is disabled. The ASR model
is cached across videos within a process.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

PathLike = Union[str, Path]

# Cache the loaded ASR model and per-language alignment models across videos.
_asr_cache: Dict[tuple, object] = {}
_align_cache: Dict[str, tuple] = {}


def _load_asr(model_size: str, device: str, compute_type: str):
    key = (model_size, device, compute_type)
    if key not in _asr_cache:
        import whisperx

        _asr_cache[key] = whisperx.load_model(
            model_size, device=device, compute_type=compute_type
        )
    return _asr_cache[key]


def transcribe_video(
    video_path: PathLike,
    model_size: str = "small",
    device: str = "cpu",
    compute_type: str = "int8",
    language: Optional[str] = None,
    batch_size: int = 16,
) -> List[Dict[str, object]]:
    """Transcribe a video to ``[{"start", "end", "text"}, ...]``.

    Timestamps are word-aligned when an alignment model is available for the
    detected language; otherwise the raw ASR segment timestamps are used. Raises
    ImportError (with a clear message) if WhisperX is not installed.
    """
    try:
        import whisperx
    except ImportError as e:  # pragma: no cover - environment-dependent
        raise ImportError(
            "WhisperX is required for transcription (B3). Install it into the "
            "active environment: `pip install whisperx`. Or run the pipeline "
            "with --no-transcript to skip audio transcription."
        ) from e

    model = _load_asr(model_size, device, compute_type)
    audio = whisperx.load_audio(str(video_path))
    result = model.transcribe(audio, batch_size=batch_size, language=language)
    segments = result.get("segments", []) or []
    lang = result.get("language", language)

    # Refine timestamps with the alignment model (best-effort).
    if segments and lang:
        try:
            if lang not in _align_cache:
                _align_cache[lang] = whisperx.load_align_model(
                    language_code=lang, device=device
                )
            align_model, metadata = _align_cache[lang]
            aligned = whisperx.align(
                segments, align_model, metadata, audio, device,
                return_char_alignments=False,
            )
            segments = aligned.get("segments", segments) or segments
        except Exception as e:  # alignment is optional; fall back to ASR times
            print(f"  [transcribe] alignment skipped ({lang}): {e}")

    lines: List[Dict[str, object]] = []
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        lines.append(
            {
                "start": round(float(seg.get("start", 0.0)), 2),
                "end": round(float(seg.get("end", 0.0)), 2),
                "text": text,
            }
        )
    return lines
