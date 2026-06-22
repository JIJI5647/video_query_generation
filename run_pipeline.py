"""v2 caption-based query-generation pipeline — main entry point.

Per video:
  segment + cut clips -> batch caption -> filter
  -> generate queries from all of the video's captions (no video; the model
     selects which captions to ground queries on)
  -> upload only the clips of the grounded segments -> verify/rewrite loop,
     where each query is checked against ONLY its own segment clip(s)
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
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent))

from emotion_query_pipeline.captioning import GeminiUploader, caption_video
from emotion_query_pipeline.caption_filter import filter_captions
from emotion_query_pipeline.omni_captioning import (
    Qwen3OmniCaptioner,
    caption_video_omni,
    omni_to_emotion_caption,
)
from emotion_query_pipeline.export import export_all
from emotion_query_pipeline.generation import generate_queries
from emotion_query_pipeline.llm_client import GeminiLLMClient
from emotion_query_pipeline.segmentation import (
    extract_segment_clips,
    grid_key,
    plan_segments,
)
from emotion_query_pipeline.stats import compute_stats
from emotion_query_pipeline.transcription import transcribe_video
from emotion_query_pipeline.validation import validate_all
from emotion_query_pipeline.video_utils import get_video_duration
from emotion_query_pipeline.workflow import PipelineResult, run_query_pipeline

_VIDEO_EXTENSIONS = (".mp4", ".avi")


def pick_videos(
    video_dir: Path, n: int, seed: int = 42, video_ids: Optional[List[str]] = None
) -> List[Path]:
    all_videos = sorted(
        p for ext in _VIDEO_EXTENSIONS for p in video_dir.glob(f"*{ext}")
    )
    if not all_videos:
        raise FileNotFoundError(
            f"No video files ({', '.join(_VIDEO_EXTENSIONS)}) found in {video_dir}"
        )
    # Pin an explicit set of video ids (stems) when given — overrides sampling.
    if video_ids:
        wanted = [v.strip() for v in video_ids if v.strip()]
        by_stem = {p.stem: p for p in all_videos}
        missing = [v for v in wanted if v not in by_stem]
        if missing:
            raise FileNotFoundError(
                f"--video-ids not found in {video_dir}: {', '.join(missing)}"
            )
        return [by_stem[v] for v in wanted]
    if n >= len(all_videos):
        return all_videos
    random.seed(seed)
    return sorted(random.sample(all_videos, n))


def main() -> None:
    parser = argparse.ArgumentParser(description="v4 caption-based query generation.")
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--num-videos", "-n", type=int, default=10)
    parser.add_argument(
        "--video-ids",
        default=None,
        help="Comma/space-separated video ids (file stems) to process exactly, "
        "in order. Overrides --num-videos/--seed sampling.",
    )
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
    parser.add_argument(
        "--segments-dir",
        default="data/processed_segments",
        help="Persistent segment-clip cache root (B4). Clips are reused across "
        "runs and not deleted afterwards.",
    )
    parser.add_argument(
        "--force-reextract",
        action="store_true",
        help="Re-cut segment clips even if cached copies exist.",
    )
    parser.add_argument(
        "--no-transcript",
        action="store_true",
        help="Skip WhisperX transcription; generation runs without dialogue text.",
    )
    parser.add_argument(
        "--whisper-model",
        default="small",
        help="WhisperX model size for transcription (e.g. tiny, base, small).",
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Skip rule-based caption filtering and feed ALL raw captions to "
        "generation, letting the model select which moments to turn into queries.",
    )
    # --- Captioning backend (Qwen3-Omni) ---
    parser.add_argument(
        "--caption-backend",
        choices=["gemini", "qwen3_omni"],
        default="gemini",
        help="Captioning backend. 'gemini' (default) uses the Gemini Files API "
        "batch path; 'qwen3_omni' uses the local-server Qwen3-Omni structured "
        "captioner (one segment per prompt).",
    )
    parser.add_argument(
        "--caption-batch-size",
        type=int,
        default=1,
        help="Segments per caption prompt for the qwen3_omni backend. Must be 1 "
        "(one segment per prompt); larger values are ignored with a warning.",
    )
    parser.add_argument(
        "--qwen-model-path",
        default="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        help="Model path/name for the qwen3_omni captioning backend.",
    )
    parser.add_argument(
        "--qwen-engine",
        choices=["vllm", "transformers"],
        default="vllm",
        help="Inference engine for the qwen3_omni backend. 'vllm' (default) is "
        "fast but needs a vLLM build matching the GPU driver's CUDA and "
        "Qwen3-Omni multimodal support; 'transformers' is a slower pure-HF "
        "fallback that only needs a working torch (use it when vLLM won't load "
        "the model as multimodal on the available CUDA/driver).",
    )
    parser.add_argument(
        "--qwen-attn-impl",
        default=None,
        help="attn_implementation for the transformers engine "
        "(e.g. flash_attention_2, sdpa, eager). Default lets HF choose.",
    )
    parser.add_argument(
        "--captions-cache-dir",
        default=None,
        help="Per-segment structured-caption cache root (qwen3_omni). Defaults to "
        "<output>/captions. Raw failed outputs go to <output>/captions_raw.",
    )
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="Skip segments that already have a valid cached caption (default).",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Disable resume; re-check every segment (still respects cache files "
        "unless --overwrite-captions is given).",
    )
    parser.add_argument(
        "--overwrite-captions",
        action="store_true",
        help="Force regeneration of every segment caption, ignoring any cache.",
    )
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    video_dir = Path(args.video_dir)
    output_dir = Path(args.output)
    # B4: segment clips live in a persistent, reusable cache (not a temp dir).
    segments_dir = Path(args.segments_dir)
    seg_subdir = grid_key(args.segment_seconds, args.stride)

    video_ids = (
        args.video_ids.replace(",", " ").split() if args.video_ids else None
    )
    selected = pick_videos(video_dir, args.num_videos, args.seed, video_ids)
    print(f"Selected {len(selected)} video(s) from {video_dir}")
    if args.caption_backend == "qwen3_omni":
        caption_desc = f"{args.qwen_model_path} ({args.qwen_engine})"
    else:
        caption_desc = args.caption_model
    print(
        f"Models — caption: {caption_desc} | generation: {args.generation_model} "
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

    # Captioning backend. The Qwen3-Omni captioner is constructed here but does
    # NOT load the model yet — the heavy vLLM/model load is lazy (first caption
    # call on the server). Generation/verification/rewrite still use Gemini.
    omni_captioner = None
    captions_cache_dir = Path(args.captions_cache_dir or (output_dir / "captions"))
    captions_raw_dir = output_dir / "captions_raw"
    if args.caption_backend == "qwen3_omni":
        omni_captioner = Qwen3OmniCaptioner(
            model_path=args.qwen_model_path,
            engine=args.qwen_engine,
            attn_implementation=args.qwen_attn_impl,
        )
        print(
            f"Caption backend — qwen3_omni ({args.qwen_model_path}) | "
            f"engine={args.qwen_engine} | batch 1 | resume={args.resume} | "
            f"overwrite={args.overwrite_captions}\n"
            f"  cache: {captions_cache_dir}"
        )

    result = PipelineResult()
    per_video_usage: List[dict] = []
    run_start = time.perf_counter()

    for i, video_path in enumerate(selected, 1):
        video_id = video_path.stem
        print(f"[{i}/{len(selected)}] {video_path.name}  (id: {video_id})")
        uploaded_segment_files: list = []
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
            # Reuse cached segment clips unless --force-reextract (B4).
            extract_segment_clips(
                video_path, video_id, segments, segments_dir,
                overwrite=args.force_reextract, subdir=seg_subdir,
            )
            print(f"  {len(segments)} segments ready (cache: {segments_dir})")

            # Steps 2-3: captions. Either the Gemini batch path or the
            # Qwen3-Omni structured path (one segment per prompt + resume cache).
            if args.caption_backend == "qwen3_omni":
                omni_caps = caption_video_omni(
                    video_id,
                    segments,
                    omni_captioner,
                    captions_cache_dir,
                    captions_raw_dir,
                    resume=args.resume,
                    overwrite=args.overwrite_captions,
                    caption_batch_size=args.caption_batch_size,
                )
                # Adapt the rich structured captions to the flat EmotionCaption
                # the rest of the pipeline (filter/generation/export) consumes.
                raw = [omni_to_emotion_caption(oc, video_id) for oc in omni_caps]
            else:
                raw = caption_video(
                    video_id, segments, client, uploader, batch_size=args.batch_size
                )
            # Step 4: rule-based filter (still recorded for the funnel stats);
            # with --no-filter we feed the model the raw captions instead and let
            # it decide which moments are worth a query.
            filtered = filter_captions(raw)
            gen_captions = raw if args.no_filter else filtered
            print(
                f"  captions: {len(raw)} raw -> {len(filtered)} kept"
                f" -> {len(gen_captions)} sent to generation"
            )

            # B3: whole-video dialogue transcript (spliced into generation only).
            transcript = None
            if not args.no_transcript:
                transcript = transcribe_video(video_path, model_size=args.whisper_model)
                print(f"  transcript: {len(transcript)} dialogue line(s)")

            # Step 5: generate queries from the video's captions + transcript
            # (no video). Grounding is by time range; segment_ids are resolved
            # internally (B1).
            gen_output = generate_queries(
                video_id, gen_captions, client, segments, transcript
            )
            print(f"  {len(gen_output.queries)} queries generated")

            # Step 6: upload only the clips of segments the queries are grounded
            # on, then verify/rewrite each query against just its own clip(s).
            if gen_output.queries:
                seg_by_id = {s.segment_id: s for s in segments}
                needed_ids = sorted(
                    {sid for q in gen_output.queries for sid in q.segment_ids}
                )
                segment_uris: dict = {}
                for sid in needed_ids:
                    seg = seg_by_id.get(sid)
                    if seg is not None and seg.clip_path:
                        f = uploader.upload(seg.clip_path)
                        uploaded_segment_files.append(f)
                        segment_uris[sid] = f.uri
                traces, ver_outs, rw_outs = run_query_pipeline(
                    video_id,
                    gen_output,
                    client,
                    segment_uris,
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
            for f in uploaded_segment_files:
                uploader.delete(f)
            # B4: segment clips are a persistent cache — do NOT delete them.
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
