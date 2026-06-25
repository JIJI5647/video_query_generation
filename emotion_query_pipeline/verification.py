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


def _decision_from_dimensions(rel: bool, ans: bool, qual: bool) -> str:
    """Decide in CODE from the three judged dimensions (not the model's word).

    - relevance OR answerability fails  -> "fail"  (not fixable by rewording)
    - both pass but query_quality fails -> "revise" (wording fix, re-verify)
    - all three pass                    -> "pass"
    A missing/non-True dimension counts as failed, so a malformed or evasive
    verifier response routes to "fail" rather than slipping through.
    """
    if not rel or not ans:
        return "fail"
    if not qual:
        return "revise"
    return "pass"


def _normalize_verification_raw(raw: dict, video_id: str, round_index: int) -> dict:
    """Fill metadata and DERIVE ``decision`` from the three dimension booleans.

    The verifier outputs only ``relevance_pass`` / ``answerability_pass`` /
    ``query_quality_pass`` (+ reason + suggested_revision); the decision is
    computed here so the routing never depends on the model's own verdict. A
    dimension that is missing or not exactly ``true`` is treated as failed.
    """
    raw.setdefault("video_id", video_id)
    raw.setdefault("round_index", round_index)
    for result in raw.get("results") or []:
        result.setdefault("video_id", video_id)
        result.setdefault("round_index", round_index)
        result.setdefault("failure_reason", "")
        result.setdefault("suggested_revision", "")
        rel = result.get("relevance_pass") is True
        ans = result.get("answerability_pass") is True
        qual = result.get("query_quality_pass") is True
        result["relevance_pass"] = rel
        result["answerability_pass"] = ans
        result["query_quality_pass"] = qual
        result["decision"] = _decision_from_dimensions(rel, ans, qual)
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


def verify_queries_many(
    video_id: str,
    queries: List[EventGroundedQuery],
    video_uris: List,
    round_index: int,
    client: BaseLLMClient,
    prompts_dir: Optional[Path] = None,
) -> VerificationBatchOutput:
    """Verify N queries in ONE batched client call (each watches its own clip(s)).

    Builds a single-query prompt per query (so each keeps its own segment clips),
    then hands the whole list to ``client.generate_json_many`` — which truly
    batches on the Qwen3-Omni engine and falls back to sequential elsewhere.
    Results are merged in input order; each single-query result has its
    ``query_id`` pinned to the trusted query so a wrong model echo can't misroute.
    """
    if not queries:
        return VerificationBatchOutput(
            video_id=video_id, round_index=round_index, results=[]
        )
    prompts = [
        build_verification_prompt(video_id, [q], round_index, prompts_dir)
        for q in queries
    ]
    raws = client.generate_json_many(
        prompts, "VerificationBatchOutput", video_uris=video_uris
    )
    results = []
    for q, raw in zip(queries, raws):
        raw = _normalize_verification_raw(raw, video_id, round_index)
        vb = VerificationBatchOutput.model_validate(raw)
        if len(vb.results) == 1:
            vb.results[0].query_id = q.query_id  # single-query prompt: trust the id
        results.extend(vb.results)
    return VerificationBatchOutput(
        video_id=video_id, round_index=round_index, results=results
    )
