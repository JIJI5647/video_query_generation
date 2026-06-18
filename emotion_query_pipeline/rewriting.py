"""Build rewrite prompts and call the LLM to fix failing queries."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

from .io_utils import load_prompt_template
from .llm_client import BaseLLMClient
from .models import EventGroundedQuery, RewriteBatchOutput, VerificationResult

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def build_rewrite_prompt(
    video_id: str,
    failing: List[Tuple[EventGroundedQuery, VerificationResult]],
    round_index: int,
    prompts_dir: Optional[Path] = None,
) -> str:
    template = load_prompt_template(
        prompts_dir or _PROMPTS_DIR, "rewrite_prompt.txt"
    )
    queries_with_feedback = [
        {
            "query_id": q.query_id,
            "query_type": q.query_type,
            "query_text": q.query_text,
            "grounding_event_description": q.grounding_event_description,
            "approximate_grounding_time": q.approximate_grounding_time,
            "target_person_or_group": q.target_person_or_group,
            "verifier_decision": vr.decision,
            "failure_reason": vr.failure_reason,
            "suggested_revision": vr.suggested_revision,
        }
        for q, vr in failing
    ]
    prompt = template
    prompt = prompt.replace("{video_id}", video_id)
    prompt = prompt.replace("{round_index}", str(round_index))
    prompt = prompt.replace(
        "{queries_with_feedback_json}",
        json.dumps(queries_with_feedback, indent=2, ensure_ascii=False),
    )
    return prompt


def _normalize_rewrite_raw(raw: dict, video_id: str, round_index: int) -> dict:
    raw.setdefault("video_id", video_id)
    raw.setdefault("round_index", round_index)
    for rewrite in raw.get("rewrites") or []:
        rewrite.setdefault("video_id", video_id)
        rewrite.setdefault("round_index", round_index)
    return raw


def rewrite_queries(
    video_id: str,
    video_uri: str,
    failing: List[Tuple[EventGroundedQuery, VerificationResult]],
    round_index: int,
    client: BaseLLMClient,
    prompts_dir: Optional[Path] = None,
) -> RewriteBatchOutput:
    if not failing:
        return RewriteBatchOutput(
            video_id=video_id, round_index=round_index, rewrites=[]
        )
    prompt = build_rewrite_prompt(video_id, failing, round_index, prompts_dir)
    raw = client.generate_json(prompt, "RewriteBatchOutput", video_uri=video_uri)
    raw = _normalize_rewrite_raw(raw, video_id, round_index)
    return RewriteBatchOutput.model_validate(raw)
