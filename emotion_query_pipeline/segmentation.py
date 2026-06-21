"""Temporal segmentation for the caption stage.

The windowing algorithm is copied from the annotation subsystem's ``grid.py``
(fully driven by ``segment_seconds``/``stride``, no integer-second assumptions)
and wrapped to emit ``Segment`` objects with ``segment_id`` = ``s001, s002, ...``.

``plan_segments`` is pure (no ffmpeg) and testable without any binaries;
``extract_segment_clips`` shells out to ffmpeg (via the copied
``clip_extractor``) to fill ``Segment.clip_path``.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Union

from .clip_extractor import extract_windows
from .models import Segment
from .schemas import TemporalWindow

PathLike = Union[str, Path]

# Tolerance for floating-point window boundaries (matches grid.py).
_EPS = 1e-6


def create_temporal_windows(
    video_id: str,
    duration: float,
    window_size: float,
    stride: float,
    include_partial_last_window: bool = True,
) -> List[TemporalWindow]:
    """Create temporal windows for a video (copied from annotation grid.py).

    Examples:
        duration=13, window_size=5, stride=5, include_partial=True
            -> 0-5, 5-10, 10-13
        duration=12.5, window_size=5, stride=2.5
            -> 0-5, 2.5-7.5, 5-10, 7.5-12.5
    """
    if window_size <= 0:
        raise ValueError(f"window_size must be > 0, got {window_size}")
    if stride <= 0:
        raise ValueError(f"stride must be > 0, got {stride}")
    if duration <= 0:
        return []

    windows: List[TemporalWindow] = []
    index = 0
    start = 0.0

    while start < duration - _EPS:
        nominal_end = start + window_size
        exceeds = nominal_end > duration + _EPS

        if exceeds:
            if not include_partial_last_window:
                break
            end = duration
        else:
            end = nominal_end

        windows.append(
            TemporalWindow(
                clip_id=f"{video_id}_clip_{index:03d}",
                index=index,
                start_time=round(start, 6),
                end_time=round(end, 6),
                window_size=float(window_size),
                stride=float(stride),
            )
        )
        index += 1

        if exceeds:
            break

        start += stride

    return windows


def _window_to_segment(window: TemporalWindow) -> Segment:
    return Segment(
        segment_id=f"s{window.index + 1:03d}",
        index=window.index,
        start_time=window.start_time,
        end_time=window.end_time,
        clip_path=None,
    )


def plan_segments(
    video_id: str,
    duration: float,
    segment_seconds: float = 5.0,
    stride: float = 5.0,
    include_partial_last_window: bool = True,
    min_segment_seconds: float = 1.0,
) -> List[Segment]:
    """Pure segmentation: duration -> ``Segment`` list (no clip extraction).

    A final partial window shorter than ``min_segment_seconds`` is dropped: such
    sliver clips (e.g. a 0.02s remainder) produce degenerate files that the Files
    API rejects, and carry no usable content. Only the last window can ever be
    partial, so dropping it keeps the remaining segment indices contiguous.
    """
    windows = create_temporal_windows(
        video_id, duration, segment_seconds, stride, include_partial_last_window
    )
    segments = [_window_to_segment(w) for w in windows]
    return [
        s for s in segments
        if round(s.end_time - s.start_time, 6) >= min_segment_seconds
    ]


def grid_key(segment_seconds: float, stride: float) -> str:
    """Cache subdir name keyed by the windowing params (B4).

    A change to ``segment_seconds`` or ``stride`` produces a different key, so a
    new grid is written into a fresh cache subdir and old clips are never reused.
    """
    return f"win{float(segment_seconds):.2f}_str{float(stride):.2f}"


def grid_key_from_segments(segments: List[Segment]) -> str:
    """Derive the cache key from a full segment list (when params aren't handy)."""
    if not segments:
        return grid_key(0.0, 0.0)
    seg_seconds = max(s.end_time - s.start_time for s in segments)
    stride = (
        segments[1].start_time - segments[0].start_time
        if len(segments) >= 2
        else seg_seconds
    )
    return grid_key(seg_seconds, stride)


def extract_segment_clips(
    video_path: PathLike,
    video_id: str,
    segments: List[Segment],
    temp_dir: PathLike,
    overwrite: bool = True,
    subdir: str = "",
) -> List[Segment]:
    """Cut a clip for each segment and fill ``Segment.clip_path`` in place.

    Rebuilds a ``TemporalWindow`` per segment so the copied ffmpeg extractor can
    run unchanged, then joins the results back by ``index``. ``subdir`` is the
    cache key (see ``grid_key``); with ``overwrite=False`` existing clips are
    reused without invoking ffmpeg.
    """
    windows = [
        TemporalWindow(
            clip_id=seg.segment_id,
            index=seg.index,
            start_time=seg.start_time,
            end_time=seg.end_time,
            window_size=round(seg.end_time - seg.start_time, 6),
            stride=round(seg.end_time - seg.start_time, 6),
        )
        for seg in segments
    ]
    extracted = extract_windows(
        video_path, video_id, windows, temp_dir, overwrite=overwrite, subdir=subdir
    )
    by_index = {clip.window.index: clip.clip_path for clip in extracted}
    for seg in segments:
        seg.clip_path = by_index.get(seg.index)
    return segments
