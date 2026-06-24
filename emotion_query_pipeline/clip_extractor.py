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
    subdir: str = "",
) -> Path:
    """Extract a single temporal window into ``temp_dir/{video_id}[/{subdir}]/``.

    Preserves video and audio. Returns the path to the written clip file. When
    ``overwrite`` is False and the clip already exists, it is reused as-is (no
    ffmpeg) — this backs the persistent segment cache (B4). ``subdir`` keys the
    cache by windowing params so a different grid never reuses old clips.

    Raises:
        RuntimeError: if ffmpeg is missing or extraction fails.
    """
    out_dir = Path(temp_dir) / video_id
    if subdir:
        out_dir = out_dir / subdir
    out_path = out_dir / clip_filename(video_id, window)

    if out_path.is_file() and not overwrite:
        return out_path

    _require_ffmpeg()
    out_dir.mkdir(parents=True, exist_ok=True)

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
    subdir: str = "",
) -> List[ExtractedClip]:
    """Extract every window for one video. Returns the extracted clips.

    Individual failures raise; callers that want batch resilience should wrap
    this per-video.
    """
    extracted: List[ExtractedClip] = []
    for window in windows:
        clip_path = extract_clip(
            video_path, window, video_id, temp_dir, overwrite=overwrite, subdir=subdir
        )
        extracted.append(ExtractedClip(window=window, clip_path=str(clip_path)))
    return extracted


def extract_frames(
    clip_path: PathLike,
    n: int = 5,
    out_dir: PathLike = None,
    overwrite: bool = False,
) -> List[str]:
    """Sample ``n`` evenly-spaced JPEG frames from a clip (for frame-based VLMs).

    Frames are written next to the clip under ``<clip_dir>/frames/<clip_stem>/``
    (or ``out_dir`` if given) as ``frame_000.jpg`` .. and reused on rerun unless
    ``overwrite``. Returns the frame paths in temporal order. Used by the
    Qwen3-VL caption backend, which reads frames rather than the whole video.
    """
    clip_path = Path(clip_path)
    n = max(1, int(n))
    frames_dir = Path(out_dir) if out_dir else clip_path.parent / "frames" / clip_path.stem
    expected = [frames_dir / f"frame_{i:03d}.jpg" for i in range(n)]
    if not overwrite and all(p.is_file() for p in expected):
        return [str(p) for p in expected]

    _require_ffmpeg()
    frames_dir.mkdir(parents=True, exist_ok=True)
    # Evenly sample n frames across the clip via the thumbnail/select filter. Use
    # fps based on probed duration so frames are spread, not clustered at the start.
    duration = _probe_duration(clip_path)
    # Place samples at the midpoints of n equal sub-intervals.
    paths: List[str] = []
    for i in range(n):
        t = duration * (i + 0.5) / n if duration > 0 else 0.0
        out_path = frames_dir / f"frame_{i:03d}.jpg"
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", f"{t:.3f}", "-i", str(clip_path),
            "-frames:v", "1", "-q:v", "2", str(out_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0 or not out_path.is_file():
            raise RuntimeError(
                f"ffmpeg failed to grab frame {i} from {clip_path}: "
                f"{result.stderr.strip()}"
            )
        paths.append(str(out_path))
    return paths


def _probe_duration(clip_path: PathLike) -> float:
    """Best-effort clip duration in seconds via ffprobe (0.0 if unknown)."""
    if shutil.which("ffprobe") is None:
        return 0.0
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(clip_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except (ValueError, AttributeError):
        return 0.0


def cleanup_clips(temp_dir: PathLike, video_id: str) -> None:
    """Delete the temp clip directory for a video (best-effort)."""
    target = Path(temp_dir) / video_id
    if target.is_dir():
        shutil.rmtree(target, ignore_errors=True)
