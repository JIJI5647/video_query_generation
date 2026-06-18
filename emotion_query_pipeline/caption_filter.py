"""Step 4: rule-based caption filtering.

Keep only captions that are confident, clearly grounded, and tied to a real
segment, so downstream query generation works from a high-precision base.
"""
from __future__ import annotations

from typing import List

from .models import EMOTION_LABEL_VALUES, EmotionCaption


def filter_captions(captions: List[EmotionCaption]) -> List[EmotionCaption]:
    """Drop weak/ungrounded captions.

    Discards a caption when any of the following holds:
    - ``confidence == "low"``
    - ``evidence_strength`` in {"weak", "ambiguous"}
    - ``segment_ids`` is empty
    - ``emotion`` is outside the eight allowed labels
    """
    kept: List[EmotionCaption] = []
    for c in captions:
        if c.confidence == "low":
            continue
        if c.evidence_strength in {"weak", "ambiguous"}:
            continue
        if not c.segment_ids:
            continue
        if c.emotion not in EMOTION_LABEL_VALUES:
            continue
        kept.append(c)
    return kept
