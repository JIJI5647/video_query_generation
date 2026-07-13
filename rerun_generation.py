"""Re-run ONLY the query-generation stage on captions from a previous run.

Skips captioning — captions are loaded back from an existing run's
JSONL artefacts. For every video it re-generates queries from all of that
video's captions (current generation prompt), re-cuts only the clips of the
grounded segments, uploads them, and runs the verify ⇄ rewrite loop (each query
checked against ONLY its own segment clip(s)), then exports a fresh output dir.

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
from emotion_query_pipeline.emotion_events import generate_emotion_events
from emotion_query_pipeline.export import export_all
from emotion_query_pipeline.generation import generate_queries
from emotion_query_pipeline.llm_client import GeminiLLMClient
from emotion_query_pipeline.models import OmniCaption, Segment
from emotion_query_pipeline.omni_captioning import Qwen3OmniCaptioner, QwenOmniLLMClient
from emotion_query_pipeline.regrounding import reground_queries
from emotion_query_pipeline.segmentation import (
    extract_segment_clips,
    grid_key_from_segments,
)
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


def _load_segments_by_video(path: Path, known_video_ids: set) -> Dict[str, list]:
    """Group ``segments.jsonl`` by video.

    ``Segment`` itself carries no ``video_id`` field (see ``models.Segment``), and
    ``segment_id`` (``s001``, ``s002``, ...) resets per video so it is NOT a safe
    cross-video key. The main pipeline's ``export.py`` injects a ``video_id`` when
    writing, but ``run_caption_generation.py`` (the batch caption-only tool) writes
    bare ``Segment`` dumps for ALL videos into one file. Handle both: use a
    per-record ``video_id`` key if present, else recover it from ``clip_path``,
    whose cache directory is always ``.../<video_id>/<grid_key>/<file>``
    (``segmentation.extract_segment_clips`` / ``clip_extractor``) — match the
    path component that's one of the video_ids we already know about (from
    ``raw_captions.jsonl``), rather than assuming a fixed path depth.
    """
    by_v: Dict[str, list] = defaultdict(list)
    for r in _read_jsonl(path):
        vid = r.get("video_id")
        if not vid and r.get("clip_path"):
            parts = set(Path(r["clip_path"]).parts)
            matches = parts & known_video_ids
            vid = next(iter(matches)) if len(matches) == 1 else None
        if vid is None:
            continue  # not attributable to a single known video -> skip
        by_v[vid].append(Segment.model_validate(r))
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
                        help="prior run dir holding raw_captions.jsonl etc.")
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-rewrites", type=int, default=3)
    parser.add_argument("--max-accepted", type=int, default=8)
    parser.add_argument("--generation-model", default="gemini-2.5-flash-lite")
    parser.add_argument("--verification-model", default="gemini-3.1-flash-lite")
    parser.add_argument("--rewrite-model", default="gemini-2.5-flash-lite")
    parser.add_argument("--temp-dir", default="temp_clips")
    parser.add_argument("--segments-dir", default="data/processed_segments")
    parser.add_argument("--force-reextract", action="store_true")
    parser.add_argument("--emotion-event-model", default=None)
    parser.add_argument(
        "--verify-rewrite-backend",
        choices=["gemini", "qwen3_omni", "nemotron", "qwen_omni_vllm"],
        default="qwen3_omni",
        help="Backend for the verification + rewrite stages (both watch the "
        "query's segment clip(s)). 'qwen3_omni' (default) watches the local clips "
        "on the shared Qwen3-Omni model (no upload, in-process transformers); "
        "'nemotron'/'qwen_omni_vllm' watch local clips via an OpenAI-compatible "
        "HTTP server (trtllm-serve / vllm serve); 'gemini' uploads clips to the "
        "Files API. Generation (query writing) always stays on Gemini.",
    )
    parser.add_argument(
        "--parallel", type=int, default=1,
        help="qwen3_omni only: queries per batched verify call (truly batched on "
        "the qwen3_omni engine; Gemini runs sequentially).",
    )
    parser.add_argument(
        "--qwen-model-path", default="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        help="Model path/name for the qwen3_omni verify/rewrite backend.",
    )
    parser.add_argument(
        "--qwen-attn-impl", default=None,
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
    # Nemotron-3-Nano-Omni (OpenAI-compatible server: trtllm-serve or vllm serve).
    parser.add_argument("--nemotron-base-url", default="http://0.0.0.0:8000/v1",
                        help="OpenAI-compatible base URL of the Nemotron server.")
    parser.add_argument(
        "--nemotron-model",
        default="nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8",
        help="Served model id (the HF repo id passed to trtllm-serve / vllm serve).",
    )
    parser.add_argument(
        "--nemotron-max-tokens", type=int, default=8192,
        help="Generation budget for the Nemotron reasoning model (raise so the "
        "chain-of-thought isn't truncated before the JSON, like Qwen Thinking).",
    )
    parser.add_argument("--nemotron-no-thinking", action="store_true",
                        help="Disable the Nemotron reasoning trace (enable_thinking=False).")
    # Qwen3-Omni served over vLLM (OpenAI-compatible server), reusing the same
    # NemotronOpenAIClient HTTP shim as the nemotron backend above.
    parser.add_argument("--qwen-vllm-base-url", default="http://0.0.0.0:8000/v1",
                        help="OpenAI-compatible base URL of the vLLM-served Qwen3-Omni server.")
    parser.add_argument("--qwen-vllm-model",
                        default="Qwen/Qwen3-Omni-30B-A3B-Instruct",
                        help="Served model id (the HF repo id passed to vllm serve).")
    parser.add_argument("--qwen-vllm-max-tokens", type=int, default=4096,
                        help="Generation budget for the vLLM-served Qwen3-Omni model.")
    parser.add_argument(
        "--qwen-vllm-thinking", action="store_true",
        help="Enable the Qwen3-Omni reasoning trace (enable_thinking=True) — only "
        "meaningful for the Thinking checkpoint. Default sends no chat_template_kwargs "
        "at all (Instruct checkpoints have no thinking mode).",
    )
    # --- Re-grounding stage (between generation and verification) ---
    parser.add_argument(
        "--regrounding",
        dest="regrounding",
        action="store_true",
        default=True,
        help="Re-select each query's grounding segment(s) with a Gemini call "
        "after generation, before verification (default on).",
    )
    parser.add_argument(
        "--no-regrounding",
        dest="regrounding",
        action="store_false",
        help="Disable the re-grounding stage.",
    )
    parser.add_argument(
        "--regrounding-scope",
        choices=["full", "window"],
        default="full",
        help="'full' (default): sees ALL of the video's captions per query. "
        "'window': restricted to +/- --regrounding-window segments around each "
        "query's original grounding.",
    )
    parser.add_argument(
        "--regrounding-window",
        type=int,
        default=2,
        help="window scope only: segments on each side of a query's original "
        "grounding offered as re-grounding candidates.",
    )
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    captions_dir = Path(args.captions_dir)
    video_dir = Path(args.video_dir)
    output_dir = Path(args.output)

    raw_captions = _load_by_video(captions_dir / "raw_captions.jsonl", OmniCaption)
    segments = _load_segments_by_video(
        captions_dir / "segments.jsonl", set(raw_captions.keys())
    )
    # No filtering — generation reads all captions and selects moments itself.
    gen_caption_source = raw_captions
    source_name = "raw_captions.jsonl"
    if not gen_caption_source:
        print(f"ERROR: no captions found in {captions_dir}/{source_name}",
              file=sys.stderr)
        sys.exit(1)

    video_ids = sorted(gen_caption_source)
    print(f"Re-generating queries for {len(video_ids)} video(s) "
          f"from {source_name} in {captions_dir}")
    if args.verify_rewrite_backend == "qwen3_omni":
        vr_desc = f"{args.qwen_model_path} (transformers)"
    elif args.verify_rewrite_backend == "nemotron":
        vr_desc = f"{args.nemotron_model} (vllm/trtllm-serve @ {args.nemotron_base_url})"
    elif args.verify_rewrite_backend == "qwen_omni_vllm":
        vr_desc = f"{args.qwen_vllm_model} (vllm serve @ {args.qwen_vllm_base_url})"
    else:
        vr_desc = f"{args.verification_model} / {args.rewrite_model}"
    print(f"Models — generation: {args.generation_model} | verify/rewrite: {vr_desc}\n")

    client = GeminiLLMClient(
        caption_model=args.generation_model,  # unused here
        generation_model=args.generation_model,
        verification_model=args.verification_model,
        rewrite_model=args.rewrite_model,
        emotion_event_model=args.emotion_event_model,
        api_key=api_key,
    )
    uploader = GeminiUploader(api_key=api_key)

    # Verify/rewrite client: shared Qwen3-Omni engine (default, watches local
    # clips, no upload), an OpenAI-compatible served backend (nemotron /
    # qwen_omni_vllm, also local clips via file:// URIs), or Gemini (uploads clips
    # to the Files API). Generation (query writing) and emotion-events always stay
    # on the Gemini `client` above.
    use_qwen_vr = args.verify_rewrite_backend == "qwen3_omni"
    # Backends that watch LOCAL clip files directly (no Gemini Files API upload).
    use_local_clips = args.verify_rewrite_backend in (
        "qwen3_omni", "nemotron", "qwen_omni_vllm",
    )
    vr_client = client
    if use_qwen_vr:
        qwen_engine = Qwen3OmniCaptioner(
            model_path=args.qwen_model_path,
            attn_implementation=args.qwen_attn_impl,
            video_reader_backend=args.qwen_video_reader_backend,
        )
        vr_client = QwenOmniLLMClient(qwen_engine)
        print(f"Verify/rewrite backend — qwen3_omni ({args.qwen_model_path}) | "
              f"engine=transformers (watches local clips, no upload)\n")
    elif args.verify_rewrite_backend == "nemotron":
        from emotion_query_pipeline.nemotron_client import NemotronOpenAIClient
        vr_client = NemotronOpenAIClient(
            base_url=args.nemotron_base_url,
            model=args.nemotron_model,
            max_tokens=args.nemotron_max_tokens,
            enable_thinking=not args.nemotron_no_thinking,
            max_workers=max(1, args.parallel),
        )
        print(f"Verify/rewrite backend — nemotron ({args.nemotron_model} @ "
              f"{args.nemotron_base_url}, max_tokens={args.nemotron_max_tokens}, "
              f"thinking={not args.nemotron_no_thinking})\n")
    elif args.verify_rewrite_backend == "qwen_omni_vllm":
        from emotion_query_pipeline.nemotron_client import NemotronOpenAIClient
        vr_client = NemotronOpenAIClient(
            base_url=args.qwen_vllm_base_url,
            model=args.qwen_vllm_model,
            max_tokens=args.qwen_vllm_max_tokens,
            enable_thinking=True if args.qwen_vllm_thinking else None,
            max_workers=max(1, args.parallel),
        )
        print(f"Verify/rewrite backend — qwen_omni_vllm ({args.qwen_vllm_model} @ "
              f"{args.qwen_vllm_base_url}, max_tokens={args.qwen_vllm_max_tokens}, "
              f"thinking={args.qwen_vllm_thinking})\n")

    print(
        f"Re-grounding — {'on' if args.regrounding else 'off'}"
        + (f" (scope={args.regrounding_scope}, window={args.regrounding_window})"
           if args.regrounding else "")
    )

    result = PipelineResult()
    per_video_usage: List[dict] = []
    run_start = time.perf_counter()

    for i, video_id in enumerate(video_ids, 1):
        print(f"[{i}/{len(video_ids)}] {video_id}")
        caps = gen_caption_source[video_id]
        uploaded_segment_files: list = []
        segments_dir = Path(args.segments_dir)
        full_segs = segments.get(video_id, [])
        seg_subdir = grid_key_from_segments(full_segs)
        v_start = time.perf_counter()
        tokens_before = client.usage_report()["total"]["total_tokens"]
        v_status = "ok"
        try:
            video_path = _find_video(video_dir, video_id)
            if video_path is None:
                raise FileNotFoundError(
                    f"no video for {video_id} in {video_dir}"
                )

            # Emotion-event stage (Gemini, text-only) from observation captions.
            event_output = generate_emotion_events(
                video_id, caps, client, full_segs
            )
            print(f"  {len(caps)} captions -> {len(event_output.events)} emotion event(s)")

            # Generate queries from observation captions + emotion events (no
            # video). Grounding is by time range; segment_ids resolved internally.
            gen_output = generate_queries(
                video_id, caps, event_output.events, client, full_segs
            )
            print(f"  {len(gen_output.queries)} queries generated")

            # Re-ground queries — a Gemini call re-selects each query's
            # grounding segment(s) from its text (no video); the original
            # generation-stage grounding is preserved in gen_* fields. Runs
            # before verification so verification checks the FINAL clip(s).
            if args.regrounding and gen_output.queries:
                gen_output.queries, rg_stats = reground_queries(
                    video_id, gen_output.queries, caps, full_segs,
                    client, scope=args.regrounding_scope,
                    window=args.regrounding_window,
                )
                result.regrounding[video_id] = rg_stats
                print(f"  reground: {rg_stats['changed']} changed, "
                      f"{rg_stats['fallback']} fell back (of {rg_stats['total']})")

            # Step 6: cut (or reuse cached) the grounded segment clips, upload
            # them, then verify/rewrite each query against just its own clip(s).
            if gen_output.queries:
                seg_by_id = {s.segment_id: s for s in full_segs}
                needed_ids = sorted(
                    {sid for q in gen_output.queries for sid in q.segment_ids}
                )
                needed_segs = [seg_by_id[sid] for sid in needed_ids if sid in seg_by_id]
                if needed_segs:
                    extract_segment_clips(
                        video_path, video_id, needed_segs, segments_dir,
                        overwrite=args.force_reextract, subdir=seg_subdir,
                    )
                segment_uris: dict = {}
                for seg in needed_segs:
                    if not seg.clip_path:
                        continue
                    if use_local_clips:
                        segment_uris[seg.segment_id] = seg.clip_path  # local path, no upload
                    else:
                        f = uploader.upload(seg.clip_path)
                        uploaded_segment_files.append(f)
                        segment_uris[seg.segment_id] = f.uri
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

            result.segments[video_id] = segments.get(video_id, [])
            result.raw_captions[video_id] = raw_captions.get(video_id, [])
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
        result.emotion_events,
        warnings,
        result.regrounding,
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
    print(f"  Captions reused      : {stats.total_raw_captions}")
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
