"""Export all pipeline outputs to disk (caption intermediates + final queries)."""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Dict, List

from .io_utils import write_jsonl
from .models import (
    EmotionCaption,
    FinalQueryRecord,
    GenerationOutput,
    PipelineStats,
    QueryTrace,
    RewriteBatchOutput,
    Segment,
    VerificationBatchOutput,
)
from .workflow import PipelineResult

_PROMPTS_SRC = Path(__file__).parent.parent / "prompts"
_PROMPT_TEMPLATES = [
    "caption_prompt.txt",
    "generation_prompt.txt",
    "verification_prompt.txt",
    "rewrite_prompt.txt",
]


# ---------------------------------------------------------------------------
# Caption-stage intermediates
# ---------------------------------------------------------------------------
def export_segments(segments: Dict[str, List[Segment]], out_path: Path) -> None:
    records = [
        {"video_id": vid, **seg.model_dump()}
        for vid, segs in segments.items()
        for seg in segs
    ]
    write_jsonl(out_path, records)


def export_captions(
    captions: Dict[str, List[EmotionCaption]], out_path: Path
) -> None:
    records = [c.model_dump() for caps in captions.values() for c in caps]
    write_jsonl(out_path, records)


# ---------------------------------------------------------------------------
# Query-stage outputs
# ---------------------------------------------------------------------------
def export_initial_queries(
    gen_outputs: Dict[str, GenerationOutput], out_path: Path
) -> None:
    write_jsonl(out_path, [g.model_dump() for g in gen_outputs.values()])


def export_verification_rounds(
    ver_outputs: Dict[str, List[VerificationBatchOutput]], out_path: Path
) -> None:
    records = [v.model_dump() for vlist in ver_outputs.values() for v in vlist]
    write_jsonl(out_path, records)


def export_rewritten_queries(
    rw_outputs: Dict[str, List[RewriteBatchOutput]], out_path: Path
) -> None:
    records = [r.model_dump() for rlist in rw_outputs.values() for r in rlist]
    write_jsonl(out_path, records)


def export_final_queries(
    video_traces: Dict[str, Dict[str, QueryTrace]], out_path: Path
) -> None:
    records = []
    for traces in video_traces.values():
        for t in traces.values():
            rec = FinalQueryRecord(
                video_id=t.video_id,
                query_id=t.query_id,
                query_type=t.query_type,
                initial_query_text=t.initial_query.query_text,
                final_query_text=t.final_query_text,
                grounding_event_description=t.grounding_event_description,
                approximate_grounding_time=t.approximate_grounding_time,
                target_person_or_group=t.target_person_or_group,
                expected_evidence=t.expected_evidence,
                segment_ids=t.segment_ids,
                rewrite_count=t.rewrite_count,
                verification_rounds=[rd.model_dump() for rd in t.verification_rounds],
                final_status=t.final_status,
            )
            records.append(rec.model_dump())
    write_jsonl(out_path, records)


def export_human_review_csv(
    video_traces: Dict[str, Dict[str, QueryTrace]], out_path: Path
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "video_id",
        "query_id",
        "query_type",
        "final_query",
        "segment_ids",
        "human_decision",
        "revised_query",
        "human_notes",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for traces in video_traces.values():
            for t in sorted(traces.values(), key=lambda x: x.query_id):
                if t.final_status != "accepted":
                    continue
                writer.writerow(
                    {
                        "video_id": t.video_id,
                        "query_id": t.query_id,
                        "query_type": t.query_type,
                        "final_query": t.final_query_text,
                        "segment_ids": " ".join(t.segment_ids),
                        "human_decision": "",
                        "revised_query": "",
                        "human_notes": "",
                    }
                )


def export_stats(stats: PipelineStats, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(stats.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8"
    )


def export_prompts_used(prompts_dir: Path) -> None:
    """Copy the four prompt templates into prompts_used/ with a version manifest."""
    prompts_dir.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, str] = {}
    for template in _PROMPT_TEMPLATES:
        src_file = _PROMPTS_SRC / template
        if not src_file.exists():
            continue
        shutil.copy2(src_file, prompts_dir / template)
        first_line = src_file.read_text(encoding="utf-8").splitlines()[0]
        if first_line.startswith("PROMPT VERSION:"):
            manifest[template] = first_line.split(":", 1)[1].strip()
        else:
            manifest[template] = template
    (prompts_dir / "prompts_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Top-level export
# ---------------------------------------------------------------------------
def export_all(
    result: PipelineResult,
    output_dir: Path,
    stats: PipelineStats | None = None,
) -> None:
    """Write every output file for the pipeline run."""
    output_dir.mkdir(parents=True, exist_ok=True)

    export_segments(result.segments, output_dir / "segments.jsonl")
    export_captions(result.raw_captions, output_dir / "raw_captions.jsonl")
    export_captions(result.filtered_captions, output_dir / "filtered_captions.jsonl")

    export_initial_queries(result.gen_outputs, output_dir / "initial_queries.jsonl")
    export_verification_rounds(result.ver_outputs, output_dir / "verification_rounds.jsonl")
    export_rewritten_queries(result.rw_outputs, output_dir / "rewritten_queries.jsonl")
    export_final_queries(result.video_traces, output_dir / "final_queries.jsonl")
    export_human_review_csv(result.video_traces, output_dir / "human_review_sheet.csv")

    if stats is not None:
        export_stats(stats, output_dir / "pipeline_stats.json")
    export_prompts_used(output_dir / "prompts_used")
