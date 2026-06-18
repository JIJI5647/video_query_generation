"""Extract temporal windows from a raw video as standalone clips with audio.

Clips are re-encoded (not stream-copied) so that arbitrary fractional
start/end timestamps from sliding windows are honoured accurately, and so
that both video and audio tracks are preserved for emotional-cue analysis.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import List, Union

from .schemas import ExtractedClip, TemporalWindow

PathLike = Union[str, Path]


def _require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg was not found on PATH. Install ffmpeg to extract clips."
        )


def clip_filename(video_id: str, window: TemporalWindow) -> str:
    """Build the temp clip filename for a window.

    Form: ``{video_id}_clip_000_start_0.00_end_5.00.mp4`` (fractional-safe).
    """
    return (
        f"{video_id}_clip_{window.index:03d}"
        f"_start_{window.start_time:.2f}"
        f"_end_{window.end_time:.2f}.mp4"
    )


def extract_clip(
    video_path: PathLike,
    window: TemporalWindow,
    video_id: str,
    temp_dir: PathLike,
    overwrite: bool = True,
) -> Path:
    """Extract a single temporal window into ``temp_dir/{video_id}/``.

    Preserves video and audio. Returns the path to the written clip file.

    Raises:
        RuntimeError: if ffmpeg is missing or extraction fails.
    """
    _require_ffmpeg()

    out_dir = Path(temp_dir) / video_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / clip_filename(video_id, window)

    if out_path.is_file() and not overwrite:
        return out_path

    duration = window.end_time - window.start_time
    if duration <= 0:
        raise RuntimeError(
            f"Refusing to extract non-positive-length window {window.clip_id} "
            f"({window.start_time}-{window.end_time})"
        )

    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-ss",
        f"{window.start_time:.3f}",
        "-i",
        str(video_path),
        "-t",
        f"{duration:.3f}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-c:a",
        "aac",
        "-avoid_negative_ts",
        "make_zero",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not out_path.is_file():
        raise RuntimeError(
            f"ffmpeg failed to extract {window.clip_id} from {video_path}: "
            f"{result.stderr.strip()}"
        )
    return out_path


def extract_windows(
    video_path: PathLike,
    video_id: str,
    windows: List[TemporalWindow],
    temp_dir: PathLike,
    overwrite: bool = True,
) -> List[ExtractedClip]:
    """Extract every window for one video. Returns the extracted clips.

    Individual failures raise; callers that want batch resilience should wrap
    this per-video.
    """
    extracted: List[ExtractedClip] = []
    for window in windows:
        clip_path = extract_clip(
            video_path, window, video_id, temp_dir, overwrite=overwrite
        )
        extracted.append(ExtractedClip(window=window, clip_path=str(clip_path)))
    return extracted


def cleanup_clips(temp_dir: PathLike, video_id: str) -> None:
    """Delete the temp clip directory for a video (best-effort)."""
    target = Path(temp_dir) / video_id
    if target.is_dir():
        shutil.rmtree(target, ignore_errors=True)
