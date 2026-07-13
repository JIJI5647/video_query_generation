"""v2 caption-based query-generation pipeline — main entry point.

Per video:
  segment + cut clips -> caption
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
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))

from emotion_query_pipeline.captioning import GeminiUploader, caption_video
from emotion_query_pipeline.omni_captioning import (
    Qwen3OmniCaptioner,
    QwenOmniLLMClient,
    caption_video_omni,
)
from emotion_query_pipeline.emotion_events import generate_emotion_events
from emotion_query_pipeline.export import export_all
from emotion_query_pipeline.generation import generate_queries
from emotion_query_pipeline.llm_client import GeminiLLMClient
from emotion_query_pipeline.regrounding import reground_queries
from emotion_query_pipeline.segmentation import (
    extract_segment_clips,
    grid_key,
    plan_segments,
)
from emotion_query_pipeline.stats import compute_stats
from emotion_query_pipeline.validation import validate_all
from emotion_query_pipeline.video_utils import get_video_duration
from emotion_query_pipeline.workflow import PipelineResult, run_query_pipeline

_VIDEO_EXTENSIONS = (".mp4", ".avi")


class StageTimer:
    """Accumulates wall-time per pipeline stage, per-video and overall.

    ``with timer.stage("captions"): ...`` adds the block's elapsed time to both
    the current video's tally and the run total. The one-time Qwen model load is
    reported separately by the captioner, so the "captions" stage here reflects
    inference (plus, on the first video, that load — called out in the summary).
    """

    def __init__(self) -> None:
        self.totals: Dict[str, float] = defaultdict(float)
        self.video: Dict[str, float] = defaultdict(float)

    def reset_video(self) -> None:
        self.video = defaultdict(float)

    @contextmanager
    def stage(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            self.video[name] += dt
            self.totals[name] += dt


def pick_videos(
    video_dir: Path,
    n: Optional[int] = None,
    seed: int = 42,
    video_ids: Optional[List[str]] = None,
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
    # No count given -> process every video. Otherwise sample n (or all if n is
    # at least the total).
    if n is None or n >= len(all_videos):
        return all_videos
    random.seed(seed)
    return sorted(random.sample(all_videos, n))


def main() -> None:
    parser = argparse.ArgumentParser(description="v4 caption-based query generation.")
    parser.add_argument("--video-dir", required=True)
    parser.add_argument(
        "--num-videos", "-n", type=int, default=None,
        help="How many videos to sample. Omit to process ALL videos in --video-dir.",
    )
    parser.add_argument(
        "--video-ids",
        default=None,
        help="Comma/space-separated video ids (file stems) to process exactly, "
        "in order. Overrides --num-videos/--seed sampling.",
    )
    parser.add_argument("--output", default="output/v2_run")
    parser.add_argument("--seed", type=int, default=42)
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
        "--emotion-event-model",
        default=None,
        help="Gemini model for the emotion-event stage. Defaults to the "
        "generation model.",
    )
    # --- Captioning backend (Qwen3-Omni) ---
    parser.add_argument(
        "--caption-backend",
        choices=["gemini", "qwen3_omni"],
        default="qwen3_omni",
        help="Captioning backend. 'qwen3_omni' (default) uses the local-server "
        "Qwen3-Omni structured captioner (one segment per prompt); 'gemini' uses "
        "the Gemini Files API batch path.",
    )
    parser.add_argument(
        "--caption-batch-size",
        type=int,
        default=1,
        help="How many segments go into ONE caption prompt (both backends). The "
        "model sees N segment clips at once and returns N captions, each mapped "
        "back to its segment_id. Default 1. Larger = fewer prompts but a bigger "
        "single prompt (and a harder mapping for the model).",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="qwen3_omni only: how many prompts to run in ONE batched model "
        "generate() call (throughput) — applies to BOTH captioning (caption "
        "prompts per call) and verify/rewrite (queries per call). Default 1. "
        "Orthogonal to --caption-batch-size; larger uses more VRAM. Truly batched "
        "on the qwen3_omni engine; the Gemini backend runs sequentially.",
    )
    parser.add_argument(
        "--qwen-model-path",
        default="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        help="Model path/name for the qwen3_omni captioning backend.",
    )
    parser.add_argument(
        "--qwen-attn-impl",
        default=None,
        help="attn_implementation for the transformers engine "
        "(e.g. flash_attention_2, sdpa, eager). Default lets HF choose.",
    )
    parser.add_argument(
        "--qwen-video-reader-backend",
        choices=["torchvision", "decord", "torchcodec"],
        default="torchvision",
        help="Force the qwen_omni_utils video reader (sets "
        "FORCE_QWENVL_VIDEO_READER). Default 'torchvision' avoids torchcodec, "
        "which often fails to load on mismatched CUDA/ffmpeg.",
    )
    parser.add_argument(
        "--verify-rewrite-backend",
        choices=["gemini", "qwen3_omni"],
        default="qwen3_omni",
        help="Backend for the verification + rewrite stages (both watch the "
        "query's segment clip(s)). 'qwen3_omni' (default) watches the local clips "
        "on the shared Qwen3-Omni model (no upload); 'gemini' uploads clips to the "
        "Files API. Generation (query writing) always stays on Gemini.",
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
    parser.add_argument(
        "--skip-verification",
        action="store_true",
        help="Stop after query generation — skip the verify/rewrite stage "
        "entirely (no clip upload/watch, no verification_results). Every "
        "video's traces end up empty, matching the existing 0-queries path.",
    )
    # --- Re-grounding stage (between generation and verification) ---
    parser.add_argument(
        "--regrounding",
        dest="regrounding",
        action="store_true",
        default=True,
        help="Re-select each query's grounding segment(s) with a Gemini call "
        "after generation, before verification (default on). The original "
        "generation-stage grounding is preserved in gen_time_range/"
        "gen_segment_ids; a missing/invalid selection falls back to it.",
    )
    parser.add_argument(
        "--no-regrounding",
        dest="regrounding",
        action="store_false",
        help="Disable the re-grounding stage; verification checks the "
        "generation stage's own grounding, as before this stage existed.",
    )
    parser.add_argument(
        "--regrounding-scope",
        choices=["full", "window"],
        default="full",
        help="'full' (default): the re-grounding call sees ALL of the video's "
        "captions for every query. 'window': restricted to +/- "
        "--regrounding-window segments around each query's original grounding.",
    )
    parser.add_argument(
        "--regrounding-window",
        type=int,
        default=2,
        help="window scope only: how many segments on each side of a query's "
        "original grounding are offered as re-grounding candidates.",
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
        caption_desc = f"{args.qwen_model_path} (transformers)"
    else:
        caption_desc = args.caption_model
    if args.verify_rewrite_backend == "qwen3_omni":
        vr_desc = f"{args.qwen_model_path} (transformers)"
    else:
        vr_desc = f"{args.verification_model} / {args.rewrite_model}"
    print(
        f"Models — caption: {caption_desc} | generation: {args.generation_model} "
        f"| verify/rewrite: {vr_desc}"
    )
    print(
        f"Segmentation — {args.segment_seconds}s window / {args.stride}s stride "
        f"| {args.caption_batch_size} segment(s)/prompt | max_accepted {args.max_accepted}\n"
    )

    client = GeminiLLMClient(
        caption_model=args.caption_model,
        generation_model=args.generation_model,
        verification_model=args.verification_model,
        rewrite_model=args.rewrite_model,
        emotion_event_model=args.emotion_event_model,
        api_key=api_key,
    )
    uploader = GeminiUploader(api_key=api_key)

    # Qwen3-Omni engine, shared across stages that need it (caption and/or
    # verify+rewrite) so the weights load ONCE. Constructed here but NOT loaded
    # yet — the heavy model load is lazy (first inference call on the server).
    use_qwen_caption = args.caption_backend == "qwen3_omni"
    use_qwen_vr = args.verify_rewrite_backend == "qwen3_omni"
    qwen_engine = None
    if use_qwen_caption or use_qwen_vr:
        qwen_engine = Qwen3OmniCaptioner(
            model_path=args.qwen_model_path,
            attn_implementation=args.qwen_attn_impl,
            video_reader_backend=args.qwen_video_reader_backend,
        )
    # Caption backend selection.
    omni_captioner = qwen_engine if use_qwen_caption else None
    captions_cache_dir = Path(args.captions_cache_dir or (output_dir / "captions"))
    captions_raw_dir = output_dir / "captions_raw"
    if use_qwen_caption:
        print(
            f"Caption backend — qwen3_omni ({args.qwen_model_path}) | "
            f"engine=transformers | {args.caption_batch_size} seg/prompt | "
            f"{args.parallel} prompt(s)/call | "
            f"resume={args.resume} | overwrite={args.overwrite_captions}\n"
            f"  cache: {captions_cache_dir}"
        )
    # Verify/rewrite client: Gemini (default) or the shared Qwen3-Omni engine.
    vr_client = QwenOmniLLMClient(qwen_engine) if use_qwen_vr else client
    if use_qwen_vr:
        print(f"Verify/rewrite backend — qwen3_omni ({args.qwen_model_path}) | "
              f"engine=transformers (watches local clips, no upload)")

    result = PipelineResult()
    per_video_usage: List[dict] = []
    timer = StageTimer()
    run_start = time.perf_counter()

    for i, video_path in enumerate(selected, 1):
        video_id = video_path.stem
        print(f"[{i}/{len(selected)}] {video_path.name}  (id: {video_id})")
        uploaded_segment_files: list = []
        timer.reset_video()
        v_start = time.perf_counter()
        tokens_before = client.usage_report()["total"]["total_tokens"]
        v_status = "ok"
        try:
            # Step 1: segment + cut clips
            print("  → [1/5] segmenting + cutting clips...", flush=True)
            with timer.stage("segment/clips"):
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

            # Step 2: OBSERVATION captions (no emotion). Qwen3-Omni structured
            # path (cached) or the Gemini batch path.
            print(f"  → [2/5] captioning {len(segments)} segment(s) "
                  f"({args.caption_backend}, parallel={args.parallel})...",
                  flush=True)
            with timer.stage("captions"):
                if args.caption_backend == "qwen3_omni":
                    raw = caption_video_omni(
                        video_id,
                        segments,
                        omni_captioner,
                        captions_cache_dir,
                        captions_raw_dir,
                        resume=args.resume,
                        overwrite=args.overwrite_captions,
                        caption_batch_size=args.caption_batch_size,
                        caption_parallel=args.parallel,
                    )
                else:
                    raw = caption_video(
                        video_id, segments, client, uploader,
                        batch_size=args.caption_batch_size,
                    )
                gen_captions = raw
            print(f"  captions: {len(raw)} observation caption(s)")

            # Step 3: emotion-event stage (Gemini, text-only) — the ONLY emotion
            # judgment. Observation captions -> EmotionEvents (8 labels).
            print("  → [3/5] inferring emotion events (Gemini)...", flush=True)
            with timer.stage("emotion_events"):
                event_output = generate_emotion_events(
                    video_id, gen_captions, client, segments
                )
            print(f"  {len(event_output.events)} emotion event(s)")

            # Step 4: generate queries from observation captions + emotion events
            # (no video). Grounding by time range; segment_ids resolved internally.
            print("  → [4/5] generating queries (Gemini)...", flush=True)
            with timer.stage("generation"):
                gen_output = generate_queries(
                    video_id, gen_captions, event_output.events, client, segments
                )
            print(f"  {len(gen_output.queries)} queries generated")

            # Step 4.5: re-ground queries — a Gemini call re-selects each
            # query's grounding segment(s) from its text (no video); the
            # original generation-stage grounding is preserved in gen_*
            # fields. Runs before verification so verification checks the
            # FINAL (re-grounded) clip(s).
            if args.regrounding and gen_output.queries:
                print(f"  → re-grounding {len(gen_output.queries)} query(ies) "
                      f"(Gemini, scope={args.regrounding_scope})...", flush=True)
                with timer.stage("reground"):
                    gen_output.queries, rg_stats = reground_queries(
                        video_id, gen_output.queries, gen_captions, segments,
                        client, scope=args.regrounding_scope,
                        window=args.regrounding_window,
                    )
                result.regrounding[video_id] = rg_stats
                print(f"  reground: {rg_stats['changed']} changed, "
                      f"{rg_stats['fallback']} fell back (of {rg_stats['total']})")

            # Step 6: make the grounded segment clips available to verify/rewrite,
            # then check each query against just its own clip(s). Gemini needs the
            # clips uploaded (URIs); Qwen3-Omni watches the local clip paths
            # directly (no upload).
            if gen_output.queries and args.skip_verification:
                print(f"  → [5/5] skipping verify/rewrite (--skip-verification): "
                      f"{len(gen_output.queries)} query(ies) left unverified")
                traces, ver_outs, rw_outs = {}, [], []
            elif gen_output.queries:
                print(f"  → [5/5] verifying + rewriting {len(gen_output.queries)} "
                      f"query(ies) ({args.verify_rewrite_backend})...", flush=True)
                with timer.stage("verify/rewrite"):
                    seg_by_id = {s.segment_id: s for s in segments}
                    needed_ids = sorted(
                        {sid for q in gen_output.queries for sid in q.segment_ids}
                    )
                    segment_uris: dict = {}
                    for sid in needed_ids:
                        seg = seg_by_id.get(sid)
                        if seg is None or not seg.clip_path:
                            continue
                        if use_qwen_vr:
                            segment_uris[sid] = seg.clip_path  # local path, no upload
                        else:
                            f = uploader.upload(seg.clip_path)
                            uploaded_segment_files.append(f)
                            segment_uris[sid] = f.uri
                    traces, ver_outs, rw_outs = run_query_pipeline(
                        video_id,
                        gen_output,
                        vr_client,
                        segment_uris,
                        max_rewrites=args.max_rewrites,
                        max_accepted=args.max_accepted,
                        verify_parallel=args.parallel,
                    )
            else:
                traces, ver_outs, rw_outs = {}, [], []

            accepted = sum(1 for t in traces.values() if t.final_status == "accepted")
            discarded = sum(1 for t in traces.values() if t.final_status == "discarded")
            print(f"  Done — {accepted} accepted, {discarded} discarded")

            result.segments[video_id] = segments
            result.raw_captions[video_id] = raw
            result.emotion_events[video_id] = event_output
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
            stage_breakdown = {k: round(v, 1) for k, v in timer.video.items()}
            per_video_usage.append(
                {
                    "video_id": video_id,
                    "status": v_status,
                    "seconds": round(v_elapsed, 1),
                    "total_tokens": v_tokens,
                    "stage_seconds": stage_breakdown,
                }
            )
            stages_str = ", ".join(
                f"{k} {v:.1f}s" for k, v in sorted(
                    timer.video.items(), key=lambda kv: kv[1], reverse=True
                )
            )
            print(f"  [usage] {v_elapsed:.1f}s, {v_tokens:,} tokens")
            if stages_str:
                print(f"  [time]  {stages_str}")

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
        result.emotion_events,
        warnings,
        result.regrounding,
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
        "stage_seconds_total": {k: round(v, 1) for k, v in timer.totals.items()},
        "per_video": per_video_usage,
    }
    (output_dir / "usage_report.json").write_text(
        json.dumps(usage_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n--- v2 Pipeline Summary ---")
    print(f"  Videos processed     : {stats.total_videos}")
    print(f"  Segments / captions  : {stats.total_segments} segs, "
          f"{stats.total_raw_captions} observation captions")
    print(f"  Emotion events       : {stats.total_emotion_events}")
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
    if timer.totals:
        total_staged = sum(timer.totals.values()) or 1.0
        print(f"  Time by stage        : (sum {total_staged/60:.1f} min; "
              f"model load reported separately above)")
        for stage, secs in sorted(
            timer.totals.items(), key=lambda kv: kv[1], reverse=True
        ):
            print(f"    - {stage:14s}: {secs:>8.1f}s  ({secs / total_staged:.0%})")
    print("\nDone.")


if __name__ == "__main__":
    main()
