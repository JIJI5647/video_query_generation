"""Validate final outputs and emit warnings for quality issues.

Carried over from v1; the only change is that the maximum-accepted cap is now a
parameter (``max_accepted``) instead of a hard-coded constant.
"""
from __future__ import annotations

import string
from collections import Counter
from typing import Dict, List

from .models import QueryTrace

_ALLOWED_QUERY_TYPES = {"explicit_event", "emotion_state", "evidence_cue"}
_ALLOWED_STATUSES = {"accepted", "discarded"}
_DEFAULT_MAX_ACCEPTED = 8
_MIN_ACCEPTED_WARN = 3
_MAX_SAME_START_PHRASE = 3


def _normalize(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def _word_overlap_ratio(a: str, b: str) -> float:
    words_a = set(_normalize(a).split())
    words_b = set(_normalize(b).split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    return len(intersection) / max(len(words_a), len(words_b))


def _leading_phrase(text: str) -> str:
    words = _normalize(text).split()
    return " ".join(words[:2]) if len(words) >= 2 else _normalize(text)


def validate_video(
    video_id: str,
    traces: Dict[str, QueryTrace],
    max_accepted: int = _DEFAULT_MAX_ACCEPTED,
) -> List[str]:
    """Run hard checks and soft warnings for one video. Returns a list of messages."""
    errors: List[str] = []
    warnings: List[str] = []

    accepted = [t for t in traces.values() if t.final_status == "accepted"]

    # --- Hard checks ---
    for t in traces.values():
        if t.final_status not in _ALLOWED_STATUSES:
            errors.append(
                f"[{video_id}] {t.query_id}: invalid final_status '{t.final_status}'"
            )
        if t.query_type not in _ALLOWED_QUERY_TYPES:
            errors.append(
                f"[{video_id}] {t.query_id}: invalid query_type '{t.query_type}'"
            )

    for t in accepted:
        if not t.final_query_text.strip():
            errors.append(
                f"[{video_id}] {t.query_id}: accepted query has empty final_query_text"
            )

    if len(accepted) > max_accepted:
        errors.append(
            f"[{video_id}] {len(accepted)} accepted queries exceed the maximum of {max_accepted}"
        )

    # Duplicate detection among accepted
    seen: set = set()
    for text in (t.final_query_text for t in accepted):
        if text in seen:
            errors.append(
                f"[{video_id}] duplicate accepted query text: '{text[:80]}'"
            )
        seen.add(text)

    # Discarded queries must not be in the accepted set
    discarded_ids = {t.query_id for t in traces.values() if t.final_status == "discarded"}
    accepted_ids = {t.query_id for t in accepted}
    overlap = discarded_ids & accepted_ids
    if overlap:
        errors.append(
            f"[{video_id}] query IDs with conflicting status: {overlap}"
        )

    # --- Soft warnings ---
    if len(accepted) < _MIN_ACCEPTED_WARN:
        warnings.append(
            f"[{video_id}] WARNING: only {len(accepted)} accepted queries "
            f"(recommended minimum: {_MIN_ACCEPTED_WARN})"
        )

    if accepted:
        type_counter = Counter(t.query_type for t in accepted)
        if len(type_counter) == 1:
            only_type = next(iter(type_counter))
            warnings.append(
                f"[{video_id}] WARNING: all accepted queries have the same "
                f"query_type '{only_type}' — low diversity"
            )

        phrase_counter = Counter(_leading_phrase(t.final_query_text) for t in accepted)
        for phrase, count in phrase_counter.items():
            if count > _MAX_SAME_START_PHRASE:
                warnings.append(
                    f"[{video_id}] WARNING: {count} accepted queries start with "
                    f"'{phrase}' — repetitive phrasing"
                )

        accepted_list = list(accepted)
        for i in range(len(accepted_list)):
            for j in range(i + 1, len(accepted_list)):
                ratio = _word_overlap_ratio(
                    accepted_list[i].final_query_text,
                    accepted_list[j].final_query_text,
                )
                if ratio >= 0.7:
                    warnings.append(
                        f"[{video_id}] WARNING: queries "
                        f"'{accepted_list[i].query_id}' and "
                        f"'{accepted_list[j].query_id}' are semantically similar "
                        f"(word overlap {ratio:.0%})"
                    )

    return errors + warnings


def validate_all(
    video_traces: Dict[str, Dict[str, QueryTrace]],
    max_accepted: int = _DEFAULT_MAX_ACCEPTED,
) -> List[str]:
    """Validate all videos and return the combined list of messages."""
    all_messages: List[str] = []
    for video_id, traces in video_traces.items():
        all_messages.extend(validate_video(video_id, traces, max_accepted))
    return all_messages
