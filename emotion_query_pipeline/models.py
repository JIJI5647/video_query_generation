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

from pydantic import BaseModel, ConfigDict, Field

# Eight fixed emotion labels for the caption stage. Tuple form for runtime
# membership checks (Literal can't be iterated directly). These eight are the
# "real" emotions; "neutral"/"unrelevant" mark a clip with no clear emotion.
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

# Schema-accepted labels for a (legacy) caption: the eight emotions plus two "no
# emotion" markers. Only the legacy Gemini caption backend still emits these; the
# new observation-only captions carry NO emotion at all.
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

# Labels an EmotionEvent may carry: exactly the eight emotion-relevant classes
# (NO "neutral"/"unrelevant"). A segment with no clear emotion-relevant evidence
# simply produces no event.
EMOTION_EVENT_LABELS = Literal[
    "angry",
    "excited",
    "fear",
    "sad",
    "surprised",
    "frustrated",
    "happy",
    "disappointed",
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
# Structured observation caption (Qwen3-VL + TimeChat / legacy Qwen3-Omni)
# ---------------------------------------------------------------------------
# A nested, OBSERVATION-ONLY caption, one per segment. It separates objective
# visual facts (``visual_objective``) from observable expression cues
# (``visual_expression``), non-transcript audio (``audio_description``) and an
# optional temporal/audio progression (``temporal_description``). It carries NO
# emotion — emotion judgment happens only in the Gemini emotion-event stage.
# ``segment_id`` / ``time_range`` are metadata only — they must not appear in any
# natural-language field. Captions are cached to disk for resume. Sub-models allow
# extra keys so a slightly richer model response never fails validation.
class OmniPerson(BaseModel):
    model_config = ConfigDict(extra="allow")

    person: str = ""
    visibility: str = ""
    position: str = ""
    action: str = ""


class OmniScene(BaseModel):
    model_config = ConfigDict(extra="allow")

    location: str = ""
    setting: str = ""


class OmniVisualObjective(BaseModel):
    """Objective visual facts only — never emotion inference (spec §5.1/§5.2)."""

    model_config = ConfigDict(extra="allow")

    people: List[OmniPerson] = Field(default_factory=list)
    scene: OmniScene = Field(default_factory=OmniScene)
    objects: List[Any] = Field(default_factory=list)
    interactions: List[Any] = Field(default_factory=list)
    key_actions: List[Any] = Field(default_factory=list)
    visibility_notes: str = ""


class OmniVisualExpression(BaseModel):
    """Observable facial / body / gaze cues for one described person."""

    model_config = ConfigDict(extra="allow")

    person: str = ""
    facial_cues: List[Any] = Field(default_factory=list)
    body_cues: List[Any] = Field(default_factory=list)
    gaze: str = ""


class OmniCaption(BaseModel):
    """One structured multimodal caption for a single video segment."""

    model_config = ConfigDict(extra="allow")

    # Metadata (filled/overridden by the backend from the Segment; never written
    # into natural-language fields).
    video_id: str = ""
    segment_id: str
    time_range: List[float] = Field(default_factory=list)
    # Content.
    # Free-form full observation narrative (used by the TimeChat-only backend,
    # which is a single captioner covering both audio and visuals). Empty for the
    # structured backends, which fill the fields below instead.
    description: str = ""
    visual_objective: OmniVisualObjective = Field(default_factory=OmniVisualObjective)
    visual_expression: List[OmniVisualExpression] = Field(default_factory=list)
    audio_description: str = ""
    # Optional non-transcript temporal/audio progression (e.g. from TimeChat).
    temporal_description: str = ""
    confidence: Literal["high", "medium", "low"] = "low"
    evidence_strength: Literal["clear", "ambiguous", "weak"] = "ambiguous"


# Top-level fields a cached OmniCaption must carry to be treated as a valid,
# resumable result. Observation-only — NO emotion. ``temporal_description`` is
# optional and intentionally NOT required.
OMNI_REQUIRED_FIELDS: tuple[str, ...] = (
    "segment_id",
    "time_range",
    "visual_objective",
    "visual_expression",
    "audio_description",
    "confidence",
    "evidence_strength",
)


# ---------------------------------------------------------------------------
# Emotion-event stage (Gemini) — the ONLY place emotion is judged
# ---------------------------------------------------------------------------
class EmotionEvent(BaseModel):
    """One emotion-relevant moment inferred by the Gemini emotion-event stage.

    Produced from observation captions only. ``emotion_label`` is restricted to
    the eight emotion-relevant classes; a moment with no clear emotion-relevant
    evidence simply yields no event. ``segment_ids`` is resolved internally from
    ``time_range`` (overlapping segments) for clip lookup / provenance.
    """

    video_id: str
    event_id: str
    emotion_label: EMOTION_EVENT_LABELS
    event_description: str
    time_range: Optional[List[float]] = None
    target_person_or_group: str = ""
    visual_evidence: List[str] = Field(default_factory=list)
    audio_evidence: List[str] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"] = "low"
    evidence_strength: Literal["clear", "ambiguous", "weak"] = "ambiguous"
    segment_ids: List[str] = Field(default_factory=list)


class EmotionEventOutput(BaseModel):
    video_id: str
    events: List[EmotionEvent] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Query stage
# ---------------------------------------------------------------------------
class GroundingEvidence(BaseModel):
    """Observable cues the model used to ground a query (debug/export only).

    Kept off the verification path — the verifier never sees these. ``visual``
    and ``audio`` come from the observation caption's observable evidence.
    """

    visual_evidence: List[str] = Field(default_factory=list)
    audio_evidence: List[str] = Field(default_factory=list)


class EventGroundedQuery(BaseModel):
    video_id: str
    query_id: str
    query_type: Literal["explicit_event", "emotion_state", "evidence_cue"]
    query_text: str
    # Legacy free-text grounding fields — now optional and no longer requested in
    # the generation prompt; superseded by structured ``grounding_evidence``.
    grounding_event_description: str = ""
    approximate_grounding_time: Optional[str] = None
    target_person_or_group: str = ""
    expected_evidence: List[str] = Field(default_factory=list)
    why_grounded: str = ""
    # v4 (B1): the model-facing grounding handle is a [start, end] time range in
    # seconds. ``segment_ids`` is resolved from it internally (overlapping
    # segments) and kept only for clip lookup / provenance — it is never shown to
    # the generation or verification models, nor to human annotators.
    time_range: Optional[List[float]] = None
    segment_ids: List[str] = Field(default_factory=list)
    # Structured, observable cues the query is grounded on (debug/export).
    grounding_evidence: Optional[GroundingEvidence] = None
    # Internal provenance: the caption ids whose segments the query covers.
    source_caption_ids: List[str] = Field(default_factory=list)


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
    target_person_or_group: str = ""
    expected_evidence: List[str] = Field(default_factory=list)
    time_range: Optional[List[float]] = None
    segment_ids: List[str] = Field(default_factory=list)
    grounding_evidence: Optional[GroundingEvidence] = None
    source_caption_ids: List[str] = Field(default_factory=list)
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
    target_person_or_group: str = ""
    expected_evidence: List[str] = Field(default_factory=list)
    time_range: Optional[List[float]] = None
    segment_ids: List[str] = Field(default_factory=list)
    grounding_evidence: Optional[GroundingEvidence] = None
    source_caption_ids: List[str] = Field(default_factory=list)
    rewrite_count: int
    verification_rounds: List[Dict[str, Any]] = Field(default_factory=list)
    final_status: Literal["accepted", "discarded"]


class PipelineStats(BaseModel):
    total_videos: int
    total_segments: int
    total_raw_captions: int
    total_emotion_events: int
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
