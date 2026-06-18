"""Video file resolution and probing helpers (ffprobe-backed)."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Union

PathLike = Union[str, Path]


def resolve_video_path(
    video_id: str,
    video_path: Optional[str],
    video_dir: PathLike,
) -> Path:
    """Resolve the raw video file path for a record.

    If ``video_path`` is provided it is used as-is; otherwise the path is
    constructed as ``{video_dir}/{video_id}.mp4``.
    """
    if video_path:
        return Path(video_path)
    return Path(video_dir) / f"{video_id}.mp4"


def video_exists(path: PathLike) -> bool:
    """Return whether the given video file exists and is a regular file."""
    p = Path(path)
    return p.is_file()


def _require_tool(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(
            f"Required tool '{name}' was not found on PATH. "
            f"Install ffmpeg (which provides {name})."
        )


def get_video_duration(path: PathLike) -> float:
    """Return the duration of a video in seconds using ffprobe.

    Raises:
        FileNotFoundError: if the video file does not exist.
        RuntimeError: if ffprobe is unavailable or fails to report a duration.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Video file not found: {p}")
    _require_tool("ffprobe")

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_entries",
        "format=duration",
        str(p),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffprobe failed for {p}: {result.stderr.strip()}"
        )

    try:
        data = json.loads(result.stdout)
        duration = float(data["format"]["duration"])
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
        raise RuntimeError(
            f"Could not parse duration from ffprobe output for {p}: {exc}"
        ) from exc

    if duration <= 0:
        raise RuntimeError(f"Non-positive duration reported for {p}: {duration}")
    return duration
