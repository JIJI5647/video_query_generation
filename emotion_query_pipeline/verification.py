"""Build verification prompts and call the LLM to check query quality."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from .io_utils import load_prompt_template
from .llm_client import BaseLLMClient
from .models import EventGroundedQuery, VerificationBatchOutput

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def build_verification_prompt(
    video_id: str,
    queries: List[EventGroundedQuery],
    round_index: int,
    prompts_dir: Optional[Path] = None,
) -> str:
    template = load_prompt_template(
        prompts_dir or _PROMPTS_DIR, "verification_prompt.txt"
    )
    # v4 (A1): the verifier sees ONLY the self-contained query_text (plus its id).
    # All caption-derived fields (query_type, grounding_event_description,
    # approximate_grounding_time, target_person_or_group, expected_evidence) are
    # withheld so a wrong caption cannot bias the judgment — the query is judged
    # purely against the clip.
    queries_payload = [
        {"query_id": q.query_id, "query_text": q.query_text}
        for q in queries
    ]
    prompt = template
    prompt = prompt.replace("{video_id}", video_id)
    prompt = prompt.replace("{round_index}", str(round_index))
    prompt = prompt.replace(
        "{queries_json}", json.dumps(queries_payload, indent=2, ensure_ascii=False)
    )
    return prompt


def _normalize_verification_raw(raw: dict, video_id: str, round_index: int) -> dict:
    """Fill fields the model sometimes omits from per-result objects."""
    raw.setdefault("video_id", video_id)
    raw.setdefault("round_index", round_index)
    bool_defaults = {
        "relevance_pass": True,
        "answerability_pass": True,
        "query_quality_pass": True,
        "is_emotion_relevant": True,
        "is_answerable_from_video": True,
        "is_grounded_in_observable_evidence": True,
        "has_hallucination": False,
        "is_english_only": True,
        "avoids_proper_nouns": True,
        "is_clear_and_unambiguous": True,
        "is_observable_not_speculative": True,
        "is_not_too_broad": True,
        "is_not_repetitive": True,
        "no_timestamp_in_query_text": True,
    }
    for result in raw.get("results") or []:
        result.setdefault("video_id", video_id)
        result.setdefault("round_index", round_index)
        result.setdefault("failure_reason", "")
        result.setdefault("suggested_revision", "")
        for key, default in bool_defaults.items():
            result.setdefault(key, default)
    return raw


def verify_queries(
    video_id: str,
    video_uri: str,
    queries: List[EventGroundedQuery],
    round_index: int,
    client: BaseLLMClient,
    prompts_dir: Optional[Path] = None,
) -> VerificationBatchOutput:
    if not queries:
        return VerificationBatchOutput(
            video_id=video_id, round_index=round_index, results=[]
        )
    prompt = build_verification_prompt(video_id, queries, round_index, prompts_dir)
    raw = client.generate_json(prompt, "VerificationBatchOutput", video_uri=video_uri)
    raw = _normalize_verification_raw(raw, video_id, round_index)
    return VerificationBatchOutput.model_validate(raw)
