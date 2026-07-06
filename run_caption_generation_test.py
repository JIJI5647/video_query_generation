"""Stage 1 of the caption-model plumbing check: caption-model inference only.

Runs a selected caption model on a clip (single pre-cut segment, or real ffmpeg-cut
5s segments), normalizes every raw output into the pipeline's ``OmniCaption``
schema, and writes the caption artefacts + ``segments.jsonl`` for the next stage
(``run_query_generation_test.py``) to pick up. Query generation and evaluation are
separate, independently-runnable stages — this script never touches Gemini.

Heavy model deps (torch / transformers / qwen_omni_utils / qwen_vl_utils / decord /
soundfile / model repos) are imported lazily inside the caption runners; importing
this script loads none of them and needs no GPU.

Examples:
    python run_caption_generation_test.py --caption-model qwen3_omni \
        --video short.mp4 --output output/caption_query_tests/qwen3_omni

    python run_caption_generation_test.py --caption-model timechat \
        --video short.mp4 --output output/caption_query_tests/timechat \
        --max-segments 3

    python run_caption_generation_test.py --caption-model qwen_audio_vl \
        --video short.mp4 --audio short.wav \
        --output output/caption_query_tests/qwen_audio_vl --max-segments 3
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent))

from emotion_query_pipeline import caption_query_test as cqt
from emotion_query_pipeline.models import OmniCaption, Segment


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run a caption model and normalize its output (stage 1/3)."
    )
    p.add_argument(
        "--caption-model", required=True, choices=cqt.supported_models(),
        help="Which caption model / model-pair to test.",
    )
    p.add_argument("--video", default=None, help="Video clip (required for AV/video models).")
    p.add_argument("--audio", default=None, help="Audio clip (required for audio+video models).")
    p.add_argument("--output", required=True, help="Output directory for caption artefacts.")
    p.add_argument("--video-id", default=None, help="Defaults to the video/audio file stem.")
    p.add_argument("--segment-seconds", type=float, default=5.0)
    p.add_argument("--max-segments", type=int, default=1)
    p.add_argument("--segment-id", default="s001", help="Segment id when using a pre-cut clip.")
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--end", type=float, default=5.0)
    p.add_argument("--caption-model-path", default=None)
    p.add_argument("--audio-model-path", default=None)
    p.add_argument("--video-model-path", default=None)
    p.add_argument("--device-map", default="auto")
    p.add_argument("--attn-impl", default=None)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    return p


def _plan_segments(
    args, spec: cqt.ModelSpec
) -> Tuple[List[Segment], Dict[str, str]]:
    """Build the segment(s) to caption, plus a per-segment audio-clip map.

    ``--max-segments 1`` (default) treats ``--video``/``--audio`` as pre-cut clips
    -> one segment spanning ``--start..--end`` with those files as-is (the audio
    map is empty; the caller falls back to ``args.audio``). ``--max-segments > 1``
    cuts real per-segment clips with the existing ffmpeg helpers (reused from the
    main pipeline): video clips for every model kind, and — for ``audio_video``-kind
    models — a matching per-segment audio slice extracted from each cut video clip
    (real per-segment audio, never a second whole-video timestamp derivation).
    Returns ``(segments, audio_clip_by_segment_id)``.
    """
    if args.max_segments <= 1:
        segment = cqt.make_segment(
            segment_id=args.segment_id, start=args.start, end=args.end,
            clip_path=args.video,
        )
        return [segment], {}

    from emotion_query_pipeline.clip_extractor import extract_audio_track
    from emotion_query_pipeline.segmentation import (
        extract_segment_clips, grid_key, plan_segments,
    )
    from emotion_query_pipeline.video_utils import get_video_duration

    video_path = Path(args.video)
    duration = get_video_duration(video_path)
    segments = plan_segments(
        args.video_id, duration, args.segment_seconds, args.segment_seconds
    )[: args.max_segments]
    subdir = grid_key(args.segment_seconds, args.segment_seconds)
    extract_segment_clips(
        video_path, args.video_id, segments,
        Path("data/processed_segments"), overwrite=False, subdir=subdir,
    )

    audio_by_segment: Dict[str, str] = {}
    if spec.requires_audio:
        audio_dir = Path("data/processed_segments") / args.video_id / subdir
        for seg in segments:
            if not seg.clip_path:
                continue
            audio_path = audio_dir / f"{Path(seg.clip_path).stem}.wav"
            extract_audio_track(seg.clip_path, audio_path, overwrite=False)
            audio_by_segment[seg.segment_id] = str(audio_path)
    return segments, audio_by_segment


# AVoCaDO's fused AV narrative and TimeChat's scene-timestamped/6-dimension
# narrative are both trusted higher than a naive free-text fallback (purpose-
# built cross-modal captioners, not just prompted generic prose) — see the
# caption_model -> confidence discussion in normalize_to_omni_caption's
# docstring. Neither is parsed into the structured fields (visual_objective /
# audio_description); their raw text passes through to Gemini as-is in
# temporal_description, which is capable of reading the embedded timestamps
# and dimension labels itself.
_DEFAULT_CONFIDENCE_BY_MODEL = {"avocado": "medium", "timechat": "medium"}


def _normalize_output(
    out: cqt.CaptionModelOutput, segment: Segment, video_id: str, caption_model: str
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
        default_confidence=_DEFAULT_CONFIDENCE_BY_MODEL.get(caption_model, "low"),
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

    if spec.non_commercial:
        print(f"NOTE: {args.caption_model} uses a NON-COMMERCIAL research-only "
              f"model ({spec.default_audio_model_path}).")

    output_dir = Path(args.output)
    args.video_id = video_id
    segments, audio_by_segment = _plan_segments(args, spec)
    print(f"[caption-gen-test] model={args.caption_model} video_id={video_id} "
          f"segments={len(segments)}")

    config = cqt.RunnerConfig(
        caption_model_path=args.caption_model_path,
        audio_model_path=args.audio_model_path,
        video_model_path=args.video_model_path,
        device_map=args.device_map,
        attn_impl=args.attn_impl,
        max_new_tokens=args.max_new_tokens,
    )

    # caption model -> raw output(s) per segment (heavy; runs on the server).
    t0 = time.perf_counter()
    captions: List[OmniCaption] = []
    raw_records: List[dict] = []
    for seg in segments:
        audio_path = audio_by_segment.get(seg.segment_id, args.audio)
        out = cqt.run_caption_model(
            args.caption_model, seg,
            video_path=(seg.clip_path or args.video), audio_path=audio_path,
            config=config,
        )
        captions.append(_normalize_output(out, seg, video_id, args.caption_model))
        raw_records.append(_raw_record(out, seg))
    caption_seconds = time.perf_counter() - t0
    print(f"[caption-gen-test] {len(captions)} normalized caption(s) in "
          f"{caption_seconds:.1f}s")

    metadata = {
        "caption_model": args.caption_model,
        "model_kind": spec.kind,
        "non_commercial": spec.non_commercial,
        "video_id": video_id,
        "video": args.video,
        "audio": args.audio,
        "num_segments": len(segments),
        "num_normalized_captions": len(captions),
        "caption_seconds": round(caption_seconds, 1),
        "model_paths": {
            "caption_model_path": args.caption_model_path or spec.default_model_path or None,
            "audio_model_path": args.audio_model_path or spec.default_audio_model_path or None,
            "video_model_path": args.video_model_path or spec.default_video_model_path or None,
        },
    }

    written = cqt.save_caption_outputs(
        output_dir,
        raw_records=raw_records,
        captions=captions,
        segments=segments,
        metadata=metadata,
    )
    print(f"\n[caption-gen-test] wrote {len(written)} file(s) to {output_dir}:")
    for name in written:
        print(f"  - {name}")
    print(f"\n[caption-gen-test] next: python run_query_generation_test.py "
          f"--captions-dir {output_dir} --output <query-output-dir>")


if __name__ == "__main__":
    main()
