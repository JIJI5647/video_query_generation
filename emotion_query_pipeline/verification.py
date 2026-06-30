"""Build verification prompts and call the LLM to check query quality."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from .io_utils import load_prompt_template
from .llm_client import BaseLLMClient
from .models import EventGroundedQuery, VerificationBatchOutput, VerificationResult

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

# Per-dimension (decomposed) verification: each dimension is judged in its OWN
# inference, instead of one call judging all three. relevance and query_quality are
# judged from the query text alone (text-only call); only answerability needs the
# clip. This routing (what to attach) is the only per-dimension thing in code — the
# prompt CONTENT for every (variant x dimension) lives entirely in
# prompts/perdim/vdim_<variant>_<slug>.txt (composed there via {{include}}), so the
# experiment design stays in the prompt files, not here.
_DIM_NEEDS_VIDEO = {
    "relevance_pass": False,
    "answerability_pass": True,
    "query_quality_pass": False,
}
_DIM_SLUG = {  # filename suffix per dimension
    "relevance_pass": "relevance",
    "answerability_pass": "answerability",
    "query_quality_pass": "query_quality",
}


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


def _build_dim_prompt(
    dim_key: str,
    video_id: str,
    query: EventGroundedQuery,
    round_index: int,
    prompts_dir: Optional[Path],
    variant: str = "p1_rule",
) -> str:
    """Load the per-dimension prompt for ``variant`` and fill the runtime fields.

    The full prompt (rule / role / few-shot / CoT for this variant + dimension) is
    authored in ``prompts/perdim/vdim_<variant>_<slug>.txt`` and composed there via
    ``{{include}}``; here we only substitute video_id / round_index / queries_json.
    """
    slug = _DIM_SLUG[dim_key]
    fname = f"perdim/vdim_{variant}_{slug}.txt"
    try:
        template = load_prompt_template(prompts_dir or _PROMPTS_DIR, fname)
    except FileNotFoundError:
        raise ValueError(
            f"no per-dimension prompt for variant '{variant}', dimension "
            f"'{dim_key}' (expected prompts/{fname})"
        )
    payload = [{"query_id": query.query_id, "query_text": query.query_text}]
    return (
        template.replace("{video_id}", video_id)
        .replace("{round_index}", str(round_index))
        .replace("{queries_json}", json.dumps(payload, indent=2, ensure_ascii=False))
    )


def _dim_value(raw: object, dim_key: str):
    """Tolerantly pull one dimension's boolean + reason from a single-dim response.

    Missing/garbled -> (False, "invalid format") so a bad response fails safe
    (consistent with the combined verifier's missing-dimension handling)."""
    if not isinstance(raw, dict):
        return False, "invalid format"
    results = raw.get("results")
    rec = results[0] if isinstance(results, list) and results else raw
    if not isinstance(rec, dict) or dim_key not in rec:
        return False, "invalid format"
    return rec.get(dim_key) is True, rec.get("failure_reason") or ""


def verify_queries_per_dimension(
    video_id: str,
    queries: List[EventGroundedQuery],
    video_uris: List,
    round_index: int,
    client: BaseLLMClient,
    prompts_dir: Optional[Path] = None,
    variant: str = "p1_rule",
) -> VerificationBatchOutput:
    """Judge each of the three dimensions in its OWN inference.

    Runs ONE ``generate_json_many`` call per dimension, the three dimensions
    SEQUENTIALLY (not batched together), so every call is modality-uniform:
    relevance / query_quality are judged from the query text only; answerability
    watches the clip. Queries are still batched within a dimension's call. Each
    dimension prompt is composed for the given strategy ``variant`` (p0..p8). The
    three booleans per query are then merged and the decision is derived in code
    with the same rule as the combined verifier.
    """
    if not queries:
        return VerificationBatchOutput(
            video_id=video_id, round_index=round_index, results=[]
        )
    vals = [dict() for _ in queries]
    reasons = [[] for _ in queries]
    # The three dimensions are judged in SEPARATE passes (one call per dimension,
    # run sequentially) so each call is modality-uniform: relevance / query_quality
    # are text-only, answerability watches the clip. Queries are still batched
    # within a dimension's call.
    for dim in _DIM_NEEDS_VIDEO:
        prompts = [
            _build_dim_prompt(dim, video_id, q, round_index, prompts_dir, variant)
            for q in queries
        ]
        uris = [(u if _DIM_NEEDS_VIDEO[dim] else None) for u in video_uris]
        raws = client.generate_json_many(
            prompts, "VerificationBatchOutput", video_uris=uris
        )
        for qi, raw in enumerate(raws):
            val, reason = _dim_value(raw, dim)
            vals[qi][dim] = val
            if not val and reason:
                reasons[qi].append(f"{dim.replace('_pass', '')}: {reason}")

    results: List[VerificationResult] = []
    for qi, q in enumerate(queries):
        rel = vals[qi].get("relevance_pass", False)
        ans = vals[qi].get("answerability_pass", False)
        qual = vals[qi].get("query_quality_pass", False)
        results.append(
            VerificationResult(
                video_id=video_id,
                query_id=q.query_id,
                round_index=round_index,
                decision=_decision_from_dimensions(rel, ans, qual),
                relevance_pass=rel,
                answerability_pass=ans,
                query_quality_pass=qual,
                failure_reason="; ".join(reasons[qi]),
                suggested_revision="",
            )
        )
    return VerificationBatchOutput(
        video_id=video_id, round_index=round_index, results=results
    )
