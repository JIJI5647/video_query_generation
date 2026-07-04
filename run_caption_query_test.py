"""Caption-model → normalized captions → EXISTING Gemini downstream → queries.

A test-oriented entry point that verifies a **new caption model can plug into the
existing pipeline**: it runs a selected caption model on a short clip, normalizes
its output into the pipeline's ``OmniCaption`` schema, then feeds those captions
through the REAL Gemini emotion-event + query-generation stages and writes the
generated queries. It is NOT a benchmark and NOT a human-quality evaluation — it
only checks that the caption→query chain runs end-to-end.

Existing ``run_pipeline.py`` behaviour is untouched. Heavy model deps (torch /
transformers / qwen_omni_utils / qwen_vl_utils / decord / soundfile / model repos)
are imported lazily inside the caption runners; importing this script loads none
of them and needs no GPU / no Gemini key.

Examples:
    python run_caption_query_test.py --caption-model qwen3_omni \
        --video short.mp4 --output output/caption_query_tests/qwen3_omni

    python run_caption_query_test.py --caption-model qwen_audio_vl \
        --video short.mp4 --audio short.wav \
        --output output/caption_query_tests/qwen_audio_vl --max-segments 1
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent))

from emotion_query_pipeline import caption_query_test as cqt
from emotion_query_pipeline.models import OmniCaption, Segment


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run a caption model → Gemini query generation integration test."
    )
    p.add_argument(
        "--caption-model", required=True, choices=cqt.supported_models(),
        help="Which caption model / model-pair to test.",
    )
    p.add_argument("--video", default=None, help="Video clip (required for AV/video models).")
    p.add_argument("--audio", default=None, help="Audio clip (required for audio+video models).")
    p.add_argument("--output", required=True, help="Output directory for all artefacts.")
    p.add_argument("--video-id", default=None, help="Defaults to the video/audio file stem.")
    p.add_argument("--segment-seconds", type=float, default=5.0)
    p.add_argument("--max-segments", type=int, default=1)
    p.add_argument("--segment-id", default="s001", help="Segment id when using a pre-cut clip.")
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--end", type=float, default=5.0)
    p.add_argument("--generation-model", default="gemini-2.5-flash-lite")
    p.add_argument(
        "--emotion-event-model", default=None,
        help="Defaults to the generation model.",
    )
    p.add_argument("--caption-model-path", default=None)
    p.add_argument("--audio-model-path", default=None)
    p.add_argument("--video-model-path", default=None)
    p.add_argument("--device-map", default="auto")
    p.add_argument("--attn-impl", default=None)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument(
        "--with-verification", action="store_true",
        help="Also run the verify/rewrite loop on Gemini and write final_queries.json.",
    )
    return p


def _plan_segments(args, spec: cqt.ModelSpec) -> List[Segment]:
    """Build the segment(s) to caption.

    ``--max-segments 1`` (default) treats ``--video`` as a pre-cut clip → one
    segment ``--segment-id`` spanning ``--start..--end`` with the clip as its
    path. ``--max-segments > 1`` is only supported for single AV/video models: the
    video is segmented and clips are cut with the existing ffmpeg helpers. Audio+
    video models require ``--max-segments 1`` (single audio file, not split).
    """
    if args.max_segments <= 1:
        return [
            cqt.make_segment(
                segment_id=args.segment_id, start=args.start, end=args.end,
                clip_path=args.video,
            )
        ]
    if spec.kind != "av":
        raise ValueError(
            f"--max-segments {args.max_segments} is only supported for single "
            f"AV/video models; {args.caption_model!r} needs paired audio+video, "
            "so re-run with --max-segments 1."
        )
    # Multi-segment: cut the real clips (reuses the pipeline's segmentation).
    from emotion_query_pipeline.segmentation import (
        extract_segment_clips, grid_key, plan_segments,
    )
    from emotion_query_pipeline.video_utils import get_video_duration

    video_path = Path(args.video)
    duration = get_video_duration(video_path)
    segments = plan_segments(
        args.video_id, duration, args.segment_seconds, args.segment_seconds
    )[: args.max_segments]
    extract_segment_clips(
        video_path, args.video_id, segments,
        Path("data/processed_segments"),
        overwrite=False, subdir=grid_key(args.segment_seconds, args.segment_seconds),
    )
    return segments


def _normalize_output(
    out: cqt.CaptionModelOutput, segment: Segment, video_id: str
) -> OmniCaption:
    if out.modality == "audio_video":
        return cqt.merge_audio_video_caption(
            out.audio_text, out.video_text, segment, video_id,
            audio_source_model=out.audio_source_model or "",
            video_source_model=out.video_source_model or "",
            source_caption_model=out.source_caption_model,
        )
    return cqt.normalize_to_omni_caption(
        out.raw_output, segment, video_id,
        source_caption_model=out.source_caption_model,
        modality=out.modality,
        audio_source_model=out.audio_source_model,
        video_source_model=out.video_source_model,
    )


def _raw_record(out: cqt.CaptionModelOutput, segment: Segment) -> dict:
    rec: dict = {"segment_id": segment.segment_id, "modality": out.modality}
    if out.modality == "audio_video":
        rec["audio_text"] = out.audio_text
        rec["video_text"] = out.video_text
        rec["audio_source_model"] = out.audio_source_model
        rec["video_source_model"] = out.video_source_model
    else:
        rec["raw_output"] = out.raw_output
    return rec


def _run_verification(video_id, gen_output, segments, api_key, args):
    """Optional: upload the grounded clips and run the verify/rewrite loop on Gemini."""
    from emotion_query_pipeline.captioning import GeminiUploader
    from emotion_query_pipeline.llm_client import GeminiLLMClient
    from emotion_query_pipeline.workflow import run_query_pipeline

    client = GeminiLLMClient(api_key=api_key)
    uploader = GeminiUploader(api_key=api_key)
    seg_by_id = {s.segment_id: s for s in segments}
    uploaded, segment_uris = [], {}
    try:
        for sid in sorted({sid for q in gen_output.queries for sid in q.segment_ids}):
            seg = seg_by_id.get(sid)
            if seg is None or not seg.clip_path:
                continue
            f = uploader.upload(seg.clip_path)
            uploaded.append(f)
            segment_uris[sid] = f.uri
        traces, _, _ = run_query_pipeline(video_id, gen_output, client, segment_uris)
    finally:
        for f in uploaded:
            uploader.delete(f)
    return [t.model_dump() for t in traces.values()]


def main() -> None:
    args = _build_arg_parser().parse_args()

    stem = Path(args.video or args.audio or "clip").stem
    video_id = args.video_id or stem

    # Input boundaries (pure) — fail fast, before touching any model.
    try:
        spec = cqt.validate_inputs(args.caption_model, args.video, args.audio)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set (needed for the Gemini downstream).",
              file=sys.stderr)
        sys.exit(1)

    if spec.non_commercial:
        print(f"NOTE: {args.caption_model} uses a NON-COMMERCIAL research-only "
              f"model ({spec.default_audio_model_path}).")

    output_dir = Path(args.output)
    segments = _plan_segments(args, spec)
    print(f"[caption-query-test] model={args.caption_model} video_id={video_id} "
          f"segments={len(segments)}")

    config = cqt.RunnerConfig(
        caption_model_path=args.caption_model_path,
        audio_model_path=args.audio_model_path,
        video_model_path=args.video_model_path,
        device_map=args.device_map,
        attn_impl=args.attn_impl,
        max_new_tokens=args.max_new_tokens,
    )

    # 1. caption model -> raw output(s) per segment (heavy; runs on the server).
    t0 = time.perf_counter()
    captions: List[OmniCaption] = []
    raw_records: List[dict] = []
    for seg in segments:
        out = cqt.run_caption_model(
            args.caption_model, seg, video_path=args.video, audio_path=args.audio,
            config=config,
        )
        captions.append(_normalize_output(out, seg, video_id))
        raw_records.append(_raw_record(out, seg))
    caption_seconds = time.perf_counter() - t0
    print(f"[caption-query-test] {len(captions)} normalized caption(s) in "
          f"{caption_seconds:.1f}s")

    # 2. downstream: REAL Gemini emotion-event + query-generation stages.
    from emotion_query_pipeline.llm_client import GeminiLLMClient
    client = GeminiLLMClient(
        generation_model=args.generation_model,
        emotion_event_model=args.emotion_event_model or args.generation_model,
        api_key=api_key,
    )
    inputs = cqt.build_downstream_inputs(video_id, captions, segments)
    downstream = cqt.run_downstream_gemini(inputs, client)
    events, generation = downstream["events"], downstream["generation"]
    warnings = list(downstream["warnings"])
    print(f"[caption-query-test] {len(events.events)} emotion event(s), "
          f"{len(generation.queries)} query(ies)")
    for w in warnings:
        print(f"  WARNING: {w}")

    # 3. optional verify/rewrite on Gemini.
    final_queries = None
    if args.with_verification and generation.queries:
        try:
            final_queries = _run_verification(
                video_id, generation, segments, api_key, args
            )
        except Exception as e:
            warnings.append(f"verification skipped: {e}")
            print(f"  WARNING: verification skipped: {e}")

    metadata = {
        "caption_model": args.caption_model,
        "model_kind": spec.kind,
        "non_commercial": spec.non_commercial,
        "video_id": video_id,
        "video": args.video,
        "audio": args.audio,
        "num_segments": len(segments),
        "num_normalized_captions": len(captions),
        "num_emotion_events": len(events.events),
        "num_generated_queries": len(generation.queries),
        "caption_seconds": round(caption_seconds, 1),
        "generation_model": args.generation_model,
        "emotion_event_model": args.emotion_event_model or args.generation_model,
        "with_verification": bool(args.with_verification),
        "warnings": warnings,
        "model_paths": {
            "caption_model_path": args.caption_model_path or spec.default_model_path or None,
            "audio_model_path": args.audio_model_path or spec.default_audio_model_path or None,
            "video_model_path": args.video_model_path or spec.default_video_model_path or None,
        },
    }

    written = cqt.save_outputs(
        output_dir,
        raw_records=raw_records,
        captions=captions,
        events=events,
        generation=generation,
        metadata=metadata,
        final_queries=final_queries,
    )
    print(f"\n[caption-query-test] wrote {len(written)} file(s) to {output_dir}:")
    for name in written:
        print(f"  - {name}")
    if not generation.queries:
        print("\n[caption-query-test] NOTE: 0 queries generated "
              "(see run_metadata.json warnings).")


if __name__ == "__main__":
    main()
