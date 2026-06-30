"""Run ONLY the verification stage on a prior run's generated queries.

Loads ``initial_queries.jsonl`` + ``segments.jsonl`` from a previous output dir
and verifies each query EXACTLY ONCE — a single verification pass, with NO
revise/rewrite loop. This is for cheaply iterating on the verification prompt or
swapping the verification model without re-captioning or re-generating queries.

Usage:
    python run_verification.py \
        --queries-dir output/test5 \
        --video-dir data/pilot_study \
        --output output/test5_verify \
        --verify-rewrite-backend qwen3_omni
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))

from emotion_query_pipeline.captioning import GeminiUploader
from emotion_query_pipeline.io_utils import write_jsonl
from emotion_query_pipeline.llm_client import GeminiLLMClient
from emotion_query_pipeline.models import GenerationOutput, Segment
from emotion_query_pipeline.omni_captioning import (
    Qwen3OmniCaptioner,
    QwenOmniLLMClient,
)
from emotion_query_pipeline.segmentation import (
    extract_segment_clips,
    grid_key_from_segments,
)
from emotion_query_pipeline.verification import verify_queries_per_dimension
from emotion_query_pipeline.workflow import _verify_per_query

_VIDEO_EXTENSIONS = (".mp4", ".avi")


def _read_jsonl(path: Path) -> List[dict]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _find_video(video_dir: Path, video_id: str):
    for ext in _VIDEO_EXTENSIONS:
        p = video_dir / f"{video_id}{ext}"
        if p.is_file():
            return p
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify previously generated queries once (no rewrite loop)."
    )
    parser.add_argument("--queries-dir", required=True,
                        help="Prior run dir holding initial_queries.jsonl + segments.jsonl.")
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--verify-rewrite-backend",
                        choices=["gemini", "qwen3_omni"], default="qwen3_omni")
    parser.add_argument("--verification-model", default="gemini-3.1-flash-lite")
    parser.add_argument(
        "--using-prompt", default=None,
        help="Path to a specific verification prompt .txt to use instead of the "
        "default prompts/verification_prompt.txt (for A/B testing prompts).",
    )
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument(
        "--per-dimension", action="store_true",
        help="Judge each of the 3 dimensions in its OWN inference (3xN calls, run "
        "in parallel) instead of one call judging all three. relevance/quality are "
        "text-only; answerability watches the clip. --using-prompt is ignored.",
    )
    parser.add_argument(
        "--variant", default="p1_rule",
        help="Strategy applied per dimension in --per-dimension mode "
        "(p0_norule, p1_rule, p2_role, p3_fewshot, p4_zscot, p5_fewshotcot, "
        "p6_rolefewshot, p7_rolecot, p8_rawcot).",
    )
    parser.add_argument("--segments-dir", default="data/processed_segments")
    parser.add_argument("--force-reextract", action="store_true")
    # Qwen3-Omni (verify backend) knobs.
    parser.add_argument("--qwen-model-path", default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    parser.add_argument("--qwen-attn-impl", default=None)
    parser.add_argument("--qwen-video-reader-backend",
                        choices=["torchvision", "decord", "torchcodec"],
                        default="torchvision")
    args = parser.parse_args()

    use_qwen_vr = args.verify_rewrite_backend == "qwen3_omni"

    queries_dir = Path(args.queries_dir)
    video_dir = Path(args.video_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    segments_dir = Path(args.segments_dir)

    # Optional custom verification prompt. build_verification_prompt always loads
    # "verification_prompt.txt" from its prompts_dir, so we stage the chosen file
    # under a temp dir with that name and pass the dir down as prompts_dir.
    prompts_dir: Optional[Path] = None
    if args.using_prompt:
        custom = Path(args.using_prompt)
        if not custom.is_file():
            print(f"ERROR: --using-prompt not found: {custom}", file=sys.stderr)
            sys.exit(1)
        prompts_dir = Path(tempfile.mkdtemp(prefix="verify_prompt_"))
        shutil.copy2(custom, prompts_dir / "verification_prompt.txt")
        # Keep a copy in the output for provenance.
        shutil.copy2(custom, output_dir / "verification_prompt_used.txt")
        first_line = custom.read_text(encoding="utf-8").splitlines()[0]
        print(f"Using custom verification prompt: {custom}\n  ({first_line})")

    # Load segments + the generated queries from the prior run.
    segments: Dict[str, List[Segment]] = collections.defaultdict(list)
    for r in _read_jsonl(queries_dir / "segments.jsonl"):
        segments[r["video_id"]].append(Segment.model_validate(r))
    gen_outputs: Dict[str, GenerationOutput] = {}
    for r in _read_jsonl(queries_dir / "initial_queries.jsonl"):
        g = GenerationOutput.model_validate(r)
        gen_outputs[g.video_id] = g
    if not gen_outputs:
        print(f"ERROR: no initial_queries.jsonl in {queries_dir}", file=sys.stderr)
        sys.exit(1)

    # Verify client: Gemini (uploads clips) or the local Qwen3-Omni engine.
    uploader = None
    if use_qwen_vr:
        engine = Qwen3OmniCaptioner(
            model_path=args.qwen_model_path,
            attn_implementation=args.qwen_attn_impl,
            video_reader_backend=args.qwen_video_reader_backend,
        )
        vr_client = QwenOmniLLMClient(engine)
        print(f"Verify backend — qwen3_omni ({args.qwen_model_path})")
    else:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("ERROR: GEMINI_API_KEY not set (needed for gemini verify).",
                  file=sys.stderr)
            sys.exit(1)
        vr_client = GeminiLLMClient(
            caption_model=args.verification_model,
            generation_model=args.verification_model,
            verification_model=args.verification_model,
            rewrite_model=args.verification_model,
            api_key=api_key,
        )
        uploader = GeminiUploader(api_key=api_key)
        print(f"Verify backend — gemini ({args.verification_model})")

    video_ids = sorted(gen_outputs)
    print(f"Verifying {len(video_ids)} video(s) from {queries_dir}\n")

    records: List[dict] = []
    decisions: collections.Counter = collections.Counter()
    run_start = time.perf_counter()

    for i, video_id in enumerate(video_ids, 1):
        gen = gen_outputs[video_id]
        full_segs = segments.get(video_id, [])
        queries = gen.queries
        print(f"[{i}/{len(video_ids)}] {video_id}  ({len(queries)} queries)")
        if not queries:
            continue
        uploaded: list = []
        try:
            seg_by_id = {s.segment_id: s for s in full_segs}
            seg_subdir = grid_key_from_segments(full_segs)
            video_path = _find_video(video_dir, video_id)
            needed_ids = sorted(
                {sid for q in queries for sid in q.segment_ids}
            )
            needed_segs = [seg_by_id[sid] for sid in needed_ids if sid in seg_by_id]
            if needed_segs and video_path is not None:
                extract_segment_clips(
                    video_path, video_id, needed_segs, segments_dir,
                    overwrite=args.force_reextract, subdir=seg_subdir,
                )
            segment_uris: dict = {}
            for seg in needed_segs:
                if not seg.clip_path:
                    continue
                if use_qwen_vr:
                    segment_uris[seg.segment_id] = seg.clip_path
                elif uploader is not None:
                    f = uploader.upload(seg.clip_path)
                    uploaded.append(f)
                    segment_uris[seg.segment_id] = f.uri

            # SINGLE verification pass — no revise/rewrite loop.
            if args.per_dimension:
                per_query_uris = [
                    [segment_uris[sid] for sid in q.segment_ids if sid in segment_uris]
                    for q in queries
                ]
                # Per-dimension prompts live under the real prompts/perdim/ dir
                # (not a --using-prompt temp stage), so pass None.
                ver_output = verify_queries_per_dimension(
                    video_id, queries, per_query_uris, 1, vr_client, None,
                    variant=args.variant, verify_parallel=args.parallel,
                )
            else:
                ver_output = _verify_per_query(
                    video_id, queries, segment_uris, 1, vr_client, prompts_dir,
                    verify_parallel=args.parallel,
                )

            qtext = {q.query_id: q for q in queries}
            for r in ver_output.results:
                decisions[r.decision] += 1
                q = qtext.get(r.query_id)
                records.append({
                    "video_id": video_id,
                    "query_id": r.query_id,
                    "query_text": q.query_text if q else "",
                    "query_type": q.query_type if q else "",
                    "decision": r.decision,
                    "relevance_pass": r.relevance_pass,
                    "answerability_pass": r.answerability_pass,
                    "query_quality_pass": r.query_quality_pass,
                    "failure_reason": r.failure_reason,
                    "suggested_revision": r.suggested_revision,
                    "time_range": q.time_range if q else None,
                    "segment_ids": q.segment_ids if q else [],
                })
            counts = collections.Counter(r.decision for r in ver_output.results)
            print(f"  {dict(counts)}")
        except Exception as e:
            print(f"  ERROR verifying {video_id}: {e} — skipping.")
        finally:
            for f in uploaded:
                uploader.delete(f)

    write_jsonl(output_dir / "verification_results.jsonl", records)

    total = sum(decisions.values()) or 1
    print(f"\n--- Verification Summary ({time.perf_counter() - run_start:.0f}s) ---")
    print(f"  Total queries verified : {sum(decisions.values())}")
    for dec in ("pass", "revise", "fail"):
        n = decisions.get(dec, 0)
        print(f"  {dec:8s}: {n:>4}  ({n / total:.0%})")
    print(f"\nWrote {output_dir / 'verification_results.jsonl'}")


if __name__ == "__main__":
    main()
