"""Batch caption generation: one caption model, many videos, model loaded ONCE.

Runs a chosen caption model over every segment of every selected video, reusing a
single loaded model across all segments/videos (see
``emotion_query_pipeline.batch_captioning``) — unlike ``run_caption_generation_test.py``,
which reloads the model per segment and is only meant for a single short clip.

Output is written in the SAME layout the main pipeline / ``rerun_generation.py``
consume, so captions produced here can be turned into queries later without
re-running the caption model:

    <output>/segments.jsonl        (one Segment per line, all videos)
    <output>/raw_captions.jsonl    (one OmniCaption per line, all videos)
    <output>/run_metadata.json
    <output>/captions/<video_id>/<segment_id>.json   (per-segment resume cache)

Then, to generate queries from these captions:
    python rerun_generation.py --captions-dir <output> --video-dir <video-dir> \
        --output <query-output-dir>

Heavy model deps are imported lazily inside the caption sessions; importing this
script loads none of them and needs no GPU.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent))

from emotion_query_pipeline import batch_captioning as bc
from emotion_query_pipeline import caption_query_test as cqt
from emotion_query_pipeline.io_utils import write_jsonl
from emotion_query_pipeline.models import OmniCaption, Segment


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Batch-caption many videos with one caption model (loaded once)."
    )
    p.add_argument(
        "--caption-model", required=True, choices=cqt.supported_models(),
        help="Which caption model / model-pair to run.",
    )
    p.add_argument("--video-dir", required=True, help="Directory of source videos.")
    p.add_argument(
        "--video-ids", default=None,
        help="Comma/space-separated video ids (file stems) to process. "
        "Omit to process every video in --video-dir.",
    )
    p.add_argument("--output", required=True, help="Output directory for caption artefacts.")
    p.add_argument("--segment-seconds", type=float, default=5.0)
    p.add_argument("--stride", type=float, default=5.0)
    p.add_argument("--min-segment-seconds", type=float, default=1.0)
    p.add_argument(
        "--max-segments-per-video", type=int, default=None,
        help="Cap segments per video (debug/smoke). Omit to caption every segment.",
    )
    p.add_argument("--segments-dir", default="data/processed_segments",
                   help="Persistent segment-clip cache root (shared with the main pipeline).")
    p.add_argument("--force-reextract", action="store_true")
    p.add_argument("--overwrite-captions", action="store_true",
                   help="Ignore the per-segment cache and re-caption everything.")
    p.add_argument("--caption-model-path", default=None)
    p.add_argument("--audio-model-path", default=None)
    p.add_argument("--video-model-path", default=None)
    p.add_argument("--device-map", default="auto")
    p.add_argument("--attn-impl", default=None)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    return p


def _find_videos(video_dir: Path, video_ids: Optional[List[str]]) -> List[Path]:
    exts = (".mp4", ".avi", ".mov", ".mkv", ".webm")
    by_stem = {p.stem: p for p in sorted(video_dir.iterdir())
               if p.suffix.lower() in exts}
    if video_ids:
        missing = [v for v in video_ids if v not in by_stem]
        if missing:
            raise ValueError(f"video id(s) not found in {video_dir}: {missing}")
        return [by_stem[v] for v in video_ids]
    return list(by_stem.values())


def _plan_video_segments(
    video_path: Path, video_id: str, args, requires_audio: bool,
) -> Tuple[List[Segment], Dict[str, str]]:
    """Cut this video's segment clips (+ per-segment audio) and return them.

    Reuses the main pipeline's ffmpeg helpers; audio slices for audio+video models
    are extracted from each cut clip (same time range as its video), so no
    separate whole-video audio input is needed.
    """
    from emotion_query_pipeline.clip_extractor import extract_audio_track
    from emotion_query_pipeline.segmentation import (
        extract_segment_clips, grid_key, plan_segments,
    )
    from emotion_query_pipeline.video_utils import get_video_duration

    duration = get_video_duration(video_path)
    segments = plan_segments(
        video_id, duration, args.segment_seconds, args.stride,
        min_segment_seconds=args.min_segment_seconds,
    )
    if args.max_segments_per_video:
        segments = segments[: args.max_segments_per_video]
    subdir = grid_key(args.segment_seconds, args.stride)
    extract_segment_clips(
        video_path, video_id, segments, Path(args.segments_dir),
        overwrite=args.force_reextract, subdir=subdir,
    )

    audio_by_segment: Dict[str, str] = {}
    if requires_audio:
        audio_dir = Path(args.segments_dir) / video_id / subdir
        for seg in segments:
            if not seg.clip_path:
                continue
            audio_path = audio_dir / f"{Path(seg.clip_path).stem}.wav"
            extract_audio_track(seg.clip_path, audio_path, overwrite=args.force_reextract)
            audio_by_segment[seg.segment_id] = str(audio_path)
    return segments, audio_by_segment


def _cache_path(output_dir: Path, video_id: str, segment_id: str) -> Path:
    return output_dir / "captions" / video_id / f"{segment_id}.json"


def main() -> None:
    args = _build_arg_parser().parse_args()

    spec = cqt.get_model_spec(args.caption_model)
    if spec.non_commercial:
        print(f"NOTE: {args.caption_model} uses a NON-COMMERCIAL research-only "
              f"model ({spec.default_audio_model_path}).")

    video_dir = Path(args.video_dir)
    video_ids = args.video_ids.replace(",", " ").split() if args.video_ids else None
    videos = _find_videos(video_dir, video_ids)
    print(f"[caption-gen] model={args.caption_model} videos={len(videos)}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = cqt.RunnerConfig(
        caption_model_path=args.caption_model_path,
        audio_model_path=args.audio_model_path,
        video_model_path=args.video_model_path,
        device_map=args.device_map,
        attn_impl=args.attn_impl,
        max_new_tokens=args.max_new_tokens,
    )

    # Load the model ONCE for the whole run.
    print(f"[caption-gen] loading {args.caption_model} (once for all segments)...",
          flush=True)
    t_load = time.perf_counter()
    session = bc.build_caption_session(args.caption_model, config)
    print(f"[caption-gen] model ready in {time.perf_counter() - t_load:.1f}s")

    all_segments: List[Segment] = []
    all_captions: List[OmniCaption] = []
    per_video: List[dict] = []
    t0 = time.perf_counter()
    try:
        for vi, video_path in enumerate(videos, 1):
            video_id = video_path.stem
            print(f"[{vi}/{len(videos)}] {video_path.name}", flush=True)
            segments, audio_by_segment = _plan_video_segments(
                video_path, video_id, args, spec.requires_audio
            )
            print(f"  {len(segments)} segment(s)")
            v_t0 = time.perf_counter()
            n_cached = 0
            for seg in segments:
                all_segments.append(seg)
                cache_file = _cache_path(output_dir, video_id, seg.segment_id)
                if cache_file.is_file() and not args.overwrite_captions:
                    cap = OmniCaption.model_validate(
                        json.loads(cache_file.read_text(encoding="utf-8"))
                    )
                    all_captions.append(cap)
                    n_cached += 1
                    continue
                audio_path = audio_by_segment.get(seg.segment_id)
                out = session.caption(
                    seg, video_path=(seg.clip_path or str(video_path)),
                    audio_path=audio_path,
                )
                cap = cqt.normalize_caption_output(
                    out, seg, video_id, args.caption_model
                )
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                cache_file.write_text(
                    json.dumps(cap.model_dump(), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                all_captions.append(cap)
            v_secs = time.perf_counter() - v_t0
            per_video.append({
                "video_id": video_id,
                "num_segments": len(segments),
                "num_cached": n_cached,
                "seconds": round(v_secs, 1),
            })
            print(f"  done in {v_secs:.1f}s ({n_cached} from cache)")
    finally:
        session.close()

    total_secs = time.perf_counter() - t0

    # Main-pipeline-compatible layout (segments.jsonl + raw_captions.jsonl).
    write_jsonl(output_dir / "segments.jsonl", all_segments)
    write_jsonl(output_dir / "raw_captions.jsonl", all_captions)
    metadata = {
        "caption_model": args.caption_model,
        "model_kind": spec.kind,
        "non_commercial": spec.non_commercial,
        "video_dir": str(video_dir),
        "num_videos": len(videos),
        "num_segments": len(all_segments),
        "num_captions": len(all_captions),
        "total_seconds": round(total_secs, 1),
        "segment_seconds": args.segment_seconds,
        "stride": args.stride,
        "per_video": per_video,
        "model_paths": {
            "caption_model_path": args.caption_model_path or spec.default_model_path or None,
            "audio_model_path": args.audio_model_path or spec.default_audio_model_path or None,
            "video_model_path": args.video_model_path or spec.default_video_model_path or None,
        },
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n[caption-gen] {len(all_captions)} caption(s) over {len(videos)} "
          f"video(s) in {total_secs / 60:.1f} min -> {output_dir}")
    print(f"[caption-gen] next: python rerun_generation.py --captions-dir {output_dir} "
          f"--video-dir {video_dir} --output <query-output-dir>")


if __name__ == "__main__":
    main()
