"""Pydantic models for the v2 caption-based query-generation pipeline.

Two model families:

1. Caption stage (new in v2): ``Segment``, ``EmotionCaption``,
   ``CaptionBatchOutput`` — the per-5s-clip emotion captions that drive
   query generation.
2. Query stage (carried over from v1, lightly extended): the
   ``EventGroundedQuery`` records ``segment_ids`` so every query traces back to
   the time segments it is grounded in. Because each segment now has exactly one
   caption, ``segment_ids`` alone identifies the grounding captions — no caption
   id is carried. The verification / rewrite models are unchanged.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

# Eight fixed emotion labels for the caption stage. Tuple form for runtime
# membership checks (Literal can't be iterated directly). These eight are the
# "real" emotions kept by the filter.
EMOTION_LABEL_VALUES: tuple[str, ...] = (
    "angry",
    "excited",
    "fear",
    "sad",
    "surprised",
    "frustrated",
    "happy",
    "disappointed",
)

# Schema-accepted labels for a caption: the eight emotions plus two "no emotion"
# markers. Every segment gets exactly one caption, so segments with no clear
# emotion are labelled here and dropped later by ``filter_captions`` (they are
# not in ``EMOTION_LABEL_VALUES``).
CAPTION_EMOTION_LABELS = Literal[
    "angry",
    "excited",
    "fear",
    "sad",
    "surprised",
    "frustrated",
    "happy",
    "disappointed",
    "neutral",
    "unrelevant",
]


# ---------------------------------------------------------------------------
# Caption stage
# ---------------------------------------------------------------------------
class Segment(BaseModel):
    """A fixed-length temporal segment of a video (s001, s002, ...)."""

    segment_id: str
    index: int
    start_time: float
    end_time: float
    clip_path: Optional[str] = None


class EmotionCaption(BaseModel):
    """One emotion caption grounded to one or more contiguous segments."""

    video_id: str
    caption_id: str
    segment_ids: List[str] = Field(default_factory=list)
    person: str
    action: str
    sound: str = "no audible cue"
    emotion: CAPTION_EMOTION_LABELS
    confidence: Literal["high", "medium", "low"]
    evidence_strength: Literal["clear", "weak", "ambiguous"]
    observable_evidence: List[str] = Field(default_factory=list)


class CaptionBatchOutput(BaseModel):
    """The captions returned for one batch of contiguous segments."""

    video_id: str
    batch_index: int
    segment_ids: List[str] = Field(default_factory=list)
    captions: List[EmotionCaption] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Query stage
# ---------------------------------------------------------------------------
class EventGroundedQuery(BaseModel):
    video_id: str
    query_id: str
    query_type: Literal["explicit_event", "emotion_state", "evidence_cue"]
    query_text: str
    grounding_event_description: str
    approximate_grounding_time: Optional[str] = None
    target_person_or_group: str
    expected_evidence: List[str] = Field(default_factory=list)
    why_grounded: str = ""
    # Provenance: query -> segments (each segment has exactly one caption)
    segment_ids: List[str] = Field(default_factory=list)


class GenerationOutput(BaseModel):
    video_id: str
    queries: List[EventGroundedQuery] = Field(default_factory=list)


class VerificationResult(BaseModel):
    video_id: str
    query_id: str
    round_index: int
    decision: Literal["pass", "fail", "revise"]
    # Three top-level criteria
    relevance_pass: bool = True
    answerability_pass: bool = True
    query_quality_pass: bool = True
    # Relevance detail
    is_emotion_relevant: bool = True
    # Answerability detail
    is_answerable_from_video: bool = True
    is_grounded_in_observable_evidence: bool = True
    has_hallucination: bool = False
    # Query-quality detail
    is_english_only: bool = True
    avoids_proper_nouns: bool = True
    is_clear_and_unambiguous: bool = True
    is_observable_not_speculative: bool = True
    is_not_too_broad: bool = True
    is_not_repetitive: bool = True
    no_timestamp_in_query_text: bool = True
    failure_reason: str = ""
    suggested_revision: str = ""


class VerificationBatchOutput(BaseModel):
    video_id: str
    round_index: int
    results: List[VerificationResult]


class RewriteRecord(BaseModel):
    video_id: str
    query_id: str
    round_index: int
    original_query_text: str
    rewritten_query_text: str
    query_type: Literal["explicit_event", "emotion_state", "evidence_cue"]
    rewrite_reason: str


class RewriteBatchOutput(BaseModel):
    video_id: str
    round_index: int
    rewrites: List[RewriteRecord]


class RoundDecision(BaseModel):
    round_index: int
    decision: Literal["pass", "fail", "revise"]
    failure_reason: str = ""


class QueryTrace(BaseModel):
    video_id: str
    query_id: str
    initial_query: EventGroundedQuery
    current_query_text: str
    final_query_text: str = ""
    query_type: Literal["explicit_event", "emotion_state", "evidence_cue"]
    grounding_event_description: str
    approximate_grounding_time: Optional[str] = None
    target_person_or_group: str
    expected_evidence: List[str] = Field(default_factory=list)
    segment_ids: List[str] = Field(default_factory=list)
    rewrite_count: int = 0
    verification_rounds: List[RoundDecision] = Field(default_factory=list)
    final_status: Literal["accepted", "discarded"] = "discarded"


class FinalQueryRecord(BaseModel):
    video_id: str
    query_id: str
    query_type: Literal["explicit_event", "emotion_state", "evidence_cue"]
    initial_query_text: str
    final_query_text: str
    grounding_event_description: str
    approximate_grounding_time: Optional[str] = None
    target_person_or_group: str
    expected_evidence: List[str] = Field(default_factory=list)
    segment_ids: List[str] = Field(default_factory=list)
    rewrite_count: int
    verification_rounds: List[Dict[str, Any]] = Field(default_factory=list)
    final_status: Literal["accepted", "discarded"]


class PipelineStats(BaseModel):
    total_videos: int
    total_segments: int
    total_raw_captions: int
    total_filtered_captions: int
    total_initial_queries: int
    total_accepted_queries: int
    total_discarded_queries: int
    average_accepted_queries_per_video: float
    emotion_distribution: Dict[str, int]
    query_type_distribution_initial: Dict[str, int]
    query_type_distribution_final_accepted: Dict[str, int]
    rewrite_count_distribution: Dict[str, int]
    pass_rate_after_initial_verification: float
    pass_rate_after_rewrites: float
    discarded_query_count: int
    diversity_warnings: List[str]
