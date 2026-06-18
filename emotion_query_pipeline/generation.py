"""Step 5: generate queries from all of a video's captions (text-only, no video).

Unlike v1, the model never sees the video here — it reads every emotion caption
extracted from the video and selects the ones worth turning into queries. Each
caption is grounded to exactly one segment, so a query's grounding is fully
described by its ``segment_ids``. We validate every query's ``segment_ids``
against the real caption segments and drop any segment_id (or whole query) the
model invented, so a query can never reference a segment without a caption.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from .io_utils import load_prompt_template
from .llm_client import BaseLLMClient
from .models import EmotionCaption, EventGroundedQuery, GenerationOutput

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def _captions_payload(captions: List[EmotionCaption]) -> list:
    return [
        {
            "segment_ids": c.segment_ids,
            "person": c.person,
            "action": c.action,
            "sound": c.sound,
            "emotion": c.emotion,
            "observable_evidence": c.observable_evidence,
        }
        for c in captions
    ]


def build_generation_prompt(
    video_id: str,
    captions: List[EmotionCaption],
    prompts_dir: Optional[Path] = None,
) -> str:
    template = load_prompt_template(
        prompts_dir or _PROMPTS_DIR, "generation_prompt.txt"
    )
    prompt = template
    prompt = prompt.replace("{video_id}", video_id)
    prompt = prompt.replace(
        "{captions_json}",
        json.dumps(_captions_payload(captions), indent=2, ensure_ascii=False),
    )
    return prompt


def generate_queries(
    video_id: str,
    captions: List[EmotionCaption],
    client: BaseLLMClient,
    prompts_dir: Optional[Path] = None,
) -> GenerationOutput:
    """Generate queries from all of a video's captions. Returns a validated GenerationOutput."""
    if not captions:
        return GenerationOutput(video_id=video_id, queries=[])

    prompt = build_generation_prompt(video_id, captions, prompts_dir)
    raw = client.generate_json(prompt, "GenerationOutput", video_uri=None)
    raw.setdefault("video_id", video_id)
    for q in raw.get("queries") or []:
        q.setdefault("video_id", video_id)

    output = GenerationOutput.model_validate(raw)
    return _validate_segment_ids(output, captions)


def _validate_segment_ids(
    output: GenerationOutput, captions: List[EmotionCaption]
) -> GenerationOutput:
    """Keep only segment_ids that exist in the captions; drop ungrounded queries.

    Guarantees every surviving query's segment_ids correspond to real caption
    segments — the model cannot ground a query to an invented segment.
    """
    valid_segments: set = {s for c in captions for s in c.segment_ids}
    kept: List[EventGroundedQuery] = []
    for q in output.queries:
        q.segment_ids = [s for s in q.segment_ids if s in valid_segments]
        if not q.segment_ids:
            continue  # drop queries with no real grounding segment
        kept.append(q)
    output.queries = kept
    return output
