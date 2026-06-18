"""v2 caption-based query-generation pipeline — main entry point.

Per video:
  segment + cut clips -> batch caption -> filter
  -> generate queries from all of the video's captions (no video; the model
     selects which captions to ground queries on)
  -> upload whole video once -> verify/rewrite loop
  -> collect; finally clean temp clips and uploaded refs.

One failing video never aborts the batch; it is logged and skipped.
Requires ffmpeg/ffprobe on PATH and GEMINI_API_KEY.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent))

from emotion_query_pipeline.captioning import GeminiUploader, caption_video
from emotion_query_pipeline.caption_filter import filter_captions
from emotion_query_pipeline.clip_extractor import cleanup_clips
from emotion_query_pipeline.export import export_all
from emotion_query_pipeline.generation import generate_queries
from emotion_query_pipeline.llm_client import GeminiLLMClient
from emotion_query_pipeline.segmentation import extract_segment_clips, plan_segments
from emotion_query_pipeline.stats import compute_stats
from emotion_query_pipeline.validation import validate_all
from emotion_query_pipeline.video_utils import get_video_duration
from emotion_query_pipeline.workflow import PipelineResult, run_query_pipeline

_VIDEO_EXTENSIONS = (".mp4", ".avi")


def pick_videos(video_dir: Path, n: int, seed: int = 42) -> List[Path]:
    all_videos = sorted(
        p for ext in _VIDEO_EXTENSIONS for p in video_dir.glob(f"*{ext}")
    )
    if not all_videos:
        raise FileNotFoundError(
            f"No video files ({', '.join(_VIDEO_EXTENSIONS)}) found in {video_dir}"
        )
    if n >= len(all_videos):
        return all_videos
    random.seed(seed)
    return sorted(random.sample(all_videos, n))


def main() -> None:
    parser = argparse.ArgumentParser(description="v2 caption-based query generation.")
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--num-videos", "-n", type=int, default=10)
    parser.add_argument("--output", default="output/v2_run")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--segment-seconds", type=float, default=5.0)
    parser.add_argument("--stride", type=float, default=5.0)
    parser.add_argument("--min-segment-seconds", type=float, default=1.0)
    parser.add_argument("--max-rewrites", type=int, default=3)
    parser.add_argument("--max-accepted", type=int, default=8)
    parser.add_argument("--caption-model", default="gemini-2.5-flash-lite")
    parser.add_argument("--generation-model", default="gemini-2.5-flash-lite")
    parser.add_argument("--verification-model", default="gemini-3.1-flash-lite")
    parser.add_argument("--rewrite-model", default="gemini-2.5-flash-lite")
    parser.add_argument("--temp-dir", default="temp_clips")
    parser.add_argument("--keep-temp-clips", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    video_dir = Path(args.video_dir)
    output_dir = Path(args.output)
    temp_dir = Path(args.temp_dir)

    selected = pick_videos(video_dir, args.num_videos, args.seed)
    print(f"Selected {len(selected)} video(s) from {video_dir}")
    print(
        f"Models — caption: {args.caption_model} | generation: {args.generation_model} "
        f"| verification: {args.verification_model} | rewrite: {args.rewrite_model}"
    )
    print(
        f"Segmentation — {args.segment_seconds}s window / {args.stride}s stride "
        f"| batch {args.batch_size} | max_accepted {args.max_accepted}\n"
    )

    client = GeminiLLMClient(
        caption_model=args.caption_model,
        generation_model=args.generation_model,
        verification_model=args.verification_model,
        rewrite_model=args.rewrite_model,
        api_key=api_key,
    )
    uploader = GeminiUploader(api_key=api_key)

    result = PipelineResult()
    per_video_usage: List[dict] = []
    run_start = time.perf_counter()

    for i, video_path in enumerate(selected, 1):
        video_id = video_path.stem
        print(f"[{i}/{len(selected)}] {video_path.name}  (id: {video_id})")
        whole_video_file = None
        v_start = time.perf_counter()
        tokens_before = client.usage_report()["total"]["total_tokens"]
        v_status = "ok"
        try:
            # Step 1: segment + cut clips
            duration = get_video_duration(video_path)
            segments = plan_segments(
                video_id, duration, args.segment_seconds, args.stride,
                min_segment_seconds=args.min_segment_seconds,
            )
            extract_segment_clips(video_path, video_id, segments, temp_dir)
            print(f"  {len(segments)} segments cut")

            # Steps 2-3: batch captions
            raw = caption_video(
                video_id, segments, client, uploader, batch_size=args.batch_size
            )
            # Step 4: filter
            filtered = filter_captions(raw)
            print(f"  captions: {len(raw)} raw -> {len(filtered)} kept")

            # Step 5: generate queries from all of the video's captions (no video)
            gen_output = generate_queries(video_id, filtered, client)
            print(f"  {len(gen_output.queries)} queries generated")

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

            result.segments[video_id] = segments
            result.raw_captions[video_id] = raw
            result.filtered_captions[video_id] = filtered
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
            if not args.keep_temp_clips:
                cleanup_clips(temp_dir, video_id)
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

    # Validate + stats + export
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

    # Token + timing report
    total_wall = time.perf_counter() - run_start
    usage = client.usage_report()
    processed = [v for v in per_video_usage if v["status"] == "ok"]
    usage_report = {
        "total_wall_seconds": round(total_wall, 1),
        "videos_attempted": len(per_video_usage),
        "videos_processed": len(processed),
        "total_tokens": usage["total"]["total_tokens"],
        "total_llm_calls": usage["total"]["calls"],
        "avg_seconds_per_processed_video": (
            round(sum(v["seconds"] for v in processed) / len(processed), 1)
            if processed else 0.0
        ),
        "avg_tokens_per_processed_video": (
            round(usage["total"]["total_tokens"] / len(processed))
            if processed else 0
        ),
        "by_stage": usage["by_stage"],
        "per_video": per_video_usage,
    }
    (output_dir / "usage_report.json").write_text(
        json.dumps(usage_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n--- v2 Pipeline Summary ---")
    print(f"  Videos processed     : {stats.total_videos}")
    print(f"  Segments / captions  : {stats.total_segments} segs, "
          f"{stats.total_raw_captions} raw -> {stats.total_filtered_captions} kept")
    print(f"  Initial queries      : {stats.total_initial_queries}")
    print(f"  Accepted queries     : {stats.total_accepted_queries}")
    print(f"  Discarded queries    : {stats.total_discarded_queries}")
    print(f"  Pass rate (round 1)  : {stats.pass_rate_after_initial_verification:.0%}")
    print(f"  Pass rate (final)    : {stats.pass_rate_after_rewrites:.0%}")
    print(f"  Wall time            : {total_wall/60:.1f} min "
          f"({usage_report['avg_seconds_per_processed_video']:.0f}s/video)")
    print(f"  Tokens               : {usage['total']['total_tokens']:,} "
          f"in {usage['total']['calls']} LLM calls")
    for stage, b in usage["by_stage"].items():
        print(f"    - {stage:12s}: {b['total_tokens']:>10,} tokens / {b['calls']} calls")
    print("\nDone.")


if __name__ == "__main__":
    main()
