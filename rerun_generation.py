"""Re-run ONLY the query-generation stage on captions from a previous run.

Skips segmentation / clip-cutting / captioning / filtering — those are loaded
back from an existing run's JSONL artefacts. For every video it re-generates
queries from all of that video's captions (current generation prompt), uploads
the whole video once, and runs the verify ⇄ rewrite loop, then exports a fresh
output dir.

Usage:
    python rerun_generation.py \
        --captions-dir output/pilot_study \
        --video-dir "../video_query_answering_demo/data/pilot study" \
        --output output/pilot_study_regen
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent))

from emotion_query_pipeline.captioning import GeminiUploader
from emotion_query_pipeline.export import export_all
from emotion_query_pipeline.generation import generate_queries
from emotion_query_pipeline.llm_client import GeminiLLMClient
from emotion_query_pipeline.models import EmotionCaption, Segment
from emotion_query_pipeline.stats import compute_stats
from emotion_query_pipeline.validation import validate_all
from emotion_query_pipeline.workflow import PipelineResult, run_query_pipeline

_VIDEO_EXTENSIONS = (".mp4", ".avi")


def _read_jsonl(path: Path) -> List[dict]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_by_video(path: Path, model) -> Dict[str, list]:
    by_v: Dict[str, list] = defaultdict(list)
    for r in _read_jsonl(path):
        by_v[r["video_id"]].append(model.model_validate(r))
    return dict(by_v)


def _find_video(video_dir: Path, video_id: str) -> Path | None:
    for ext in _VIDEO_EXTENSIONS:
        p = video_dir / f"{video_id}{ext}"
        if p.is_file():
            return p
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-run query generation from previously generated captions."
    )
    parser.add_argument("--captions-dir", required=True,
                        help="prior run dir holding filtered_captions.jsonl etc.")
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-rewrites", type=int, default=3)
    parser.add_argument("--max-accepted", type=int, default=8)
    parser.add_argument("--generation-model", default="gemini-2.5-flash-lite")
    parser.add_argument("--verification-model", default="gemini-3.1-flash-lite")
    parser.add_argument("--rewrite-model", default="gemini-2.5-flash-lite")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    captions_dir = Path(args.captions_dir)
    video_dir = Path(args.video_dir)
    output_dir = Path(args.output)

    segments = _load_by_video(captions_dir / "segments.jsonl", Segment)
    raw_captions = _load_by_video(captions_dir / "raw_captions.jsonl", EmotionCaption)
    filtered_captions = _load_by_video(
        captions_dir / "filtered_captions.jsonl", EmotionCaption
    )
    if not filtered_captions:
        print(f"ERROR: no captions found in {captions_dir}/filtered_captions.jsonl",
              file=sys.stderr)
        sys.exit(1)

    video_ids = sorted(filtered_captions)
    print(f"Re-generating queries for {len(video_ids)} video(s) "
          f"from captions in {captions_dir}")
    print(f"Models — generation: {args.generation_model} "
          f"| verification: {args.verification_model} | rewrite: {args.rewrite_model}\n")

    client = GeminiLLMClient(
        caption_model=args.generation_model,  # unused here
        generation_model=args.generation_model,
        verification_model=args.verification_model,
        rewrite_model=args.rewrite_model,
        api_key=api_key,
    )
    uploader = GeminiUploader(api_key=api_key)

    result = PipelineResult()
    per_video_usage: List[dict] = []
    run_start = time.perf_counter()

    for i, video_id in enumerate(video_ids, 1):
        print(f"[{i}/{len(video_ids)}] {video_id}")
        caps = filtered_captions[video_id]
        whole_video_file = None
        v_start = time.perf_counter()
        tokens_before = client.usage_report()["total"]["total_tokens"]
        v_status = "ok"
        try:
            video_path = _find_video(video_dir, video_id)
            if video_path is None:
                raise FileNotFoundError(
                    f"no video for {video_id} in {video_dir}"
                )

            # Step 5: generate queries from all of the video's captions (no video)
            gen_output = generate_queries(video_id, caps, client)
            print(f"  {len(caps)} captions -> {len(gen_output.queries)} queries generated")

            # Step 6: upload whole video once, then verify/rewrite
            if gen_output.queries:
                whole_video_file = uploader.upload(str(video_path))
                traces, ver_outs, rw_outs = run_query_pipeline(
                    video_id,
                    whole_video_file.uri,
                    gen_output,
                    client,
                    max_rewrites=args.max_rewrites,
                    max_accepted=args.max_accepted,
                )
            else:
                traces, ver_outs, rw_outs = {}, [], []

            accepted = sum(1 for t in traces.values() if t.final_status == "accepted")
            discarded = sum(1 for t in traces.values() if t.final_status == "discarded")
            print(f"  Done — {accepted} accepted, {discarded} discarded")

            result.segments[video_id] = segments.get(video_id, [])
            result.raw_captions[video_id] = raw_captions.get(video_id, [])
            result.filtered_captions[video_id] = caps
            result.gen_outputs[video_id] = gen_output
            result.video_traces[video_id] = traces
            result.ver_outputs[video_id] = ver_outs
            result.rw_outputs[video_id] = rw_outs
        except Exception as e:
            v_status = "skipped"
            print(f"  ERROR processing {video_id}: {e} — skipping.")
        finally:
            if whole_video_file is not None:
                uploader.delete(whole_video_file)
            v_elapsed = time.perf_counter() - v_start
            v_tokens = client.usage_report()["total"]["total_tokens"] - tokens_before
            per_video_usage.append(
                {
                    "video_id": video_id,
                    "status": v_status,
                    "seconds": round(v_elapsed, 1),
                    "total_tokens": v_tokens,
                }
            )
            print(f"  [usage] {v_elapsed:.1f}s, {v_tokens:,} tokens")

    if not result.video_traces:
        print("\nERROR: No videos produced queries.", file=sys.stderr)
        sys.exit(1)

    warnings = validate_all(result.video_traces, max_accepted=args.max_accepted)
    for w in warnings:
        print(f"  {w}")
    result.validation_warnings = warnings

    stats = compute_stats(
        result.video_traces,
        result.gen_outputs,
        result.ver_outputs,
        result.segments,
        result.raw_captions,
        result.filtered_captions,
        warnings,
    )

    print(f"\nExporting to: {output_dir}")
    export_all(result, output_dir, stats)

    total_wall = time.perf_counter() - run_start
    usage = client.usage_report()
    processed = [v for v in per_video_usage if v["status"] == "ok"]
    usage_report = {
        "total_wall_seconds": round(total_wall, 1),
        "videos_attempted": len(per_video_usage),
        "videos_processed": len(processed),
        "total_tokens": usage["total"]["total_tokens"],
        "total_llm_calls": usage["total"]["calls"],
        "by_stage": usage["by_stage"],
        "per_video": per_video_usage,
    }
    (output_dir / "usage_report.json").write_text(
        json.dumps(usage_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n--- Re-generation Summary ---")
    print(f"  Videos processed     : {stats.total_videos}")
    print(f"  Captions reused      : {stats.total_filtered_captions}")
    print(f"  Initial queries      : {stats.total_initial_queries}")
    print(f"  Accepted queries     : {stats.total_accepted_queries}")
    print(f"  Discarded queries    : {stats.total_discarded_queries}")
    print(f"  Pass rate (round 1)  : {stats.pass_rate_after_initial_verification:.0%}")
    print(f"  Pass rate (final)    : {stats.pass_rate_after_rewrites:.0%}")
    print(f"  Wall time            : {total_wall/60:.1f} min")
    print(f"  Tokens               : {usage['total']['total_tokens']:,} "
          f"in {usage['total']['calls']} LLM calls")
    print("\nDone.")


if __name__ == "__main__":
    main()
