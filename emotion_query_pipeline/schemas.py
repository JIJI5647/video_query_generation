"""Minimal temporal-window schemas reused by the clip extractor.

Copied (trimmed) from the annotation subsystem so the v2 query-generation
project is self-contained. Only the two models needed by ``clip_extractor.py``
and ``segmentation.py`` are kept here.
"""
from __future__ import annotations

from pydantic import BaseModel


class TemporalWindow(BaseModel):
    """A temporal window over the video timeline (floating-point seconds)."""

    clip_id: str
    index: int
    start_time: float
    end_time: float
    window_size: float
    stride: float

    @property
    def duration(self) -> float:
        return round(self.end_time - self.start_time, 6)


class ExtractedClip(BaseModel):
    """A temporal window paired with the path to its extracted clip file."""

    window: TemporalWindow
    clip_path: str
