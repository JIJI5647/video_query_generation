"""Reference-free MLLM-as-judge for caption quality + emotion-event quality.

Scores caption models (e.g. nemotron_omni, qwen3_omni, timechat) on sampled 5s
clips, with a MULTIMODAL Gemini judge that WATCHES the clip (via the pipeline's
existing Files-API upload path — ``captioning.GeminiUploader`` +
``llm_client.GeminiLLMClient``). Reference-free throughout: no ground-truth
caption or label is used, only the clip itself. Design (rubric, bias
mitigations, JSON schema) lives in ``prompts/judge/*.txt``; sampling / prompt
filling / verdict parsing / aggregation are pure Python in
``emotion_query_pipeline/caption_judge.py`` (independently unit-tested, no
Gemini/GPU needed).

Two independent judging passes per model:
  1. CAPTION quality — one call per sampled (clip, caption), pointwise, all
     five dimensions (faithfulness, hallucination, coverage, fluency,
     emotion_leakage_ok) returned together.
  2. EMOTION-EVENT quality — one call per sampled (clip, event), pointwise,
     three dimensions (cue_sufficiency, cue_grounded, label_agreement).

Resumable: each verdict is appended to its model's JSONL as soon as it's
scored, and already-scored (video_id, item_id) pairs are skipped on a re-run.
Robust: a judge call that raises, or returns unparseable JSON, is recorded
with its error and the run continues — it never aborts the whole batch.

Usage:
    python judge_captions.py --models nemotron_omni,qwen3_omni,timechat \\
        --captions-root output/caption_gen_unified19 \\
        --events-root output/eval_unified19 \\
        --n-captions 100 --n-events 50 \\
        --judge-model gemini-2.5-flash \\
        --output output/caption_judge

Smoke test (tiny, real Gemini calls):
    python judge_captions.py --models nemotron_omni,qwen3_omni,timechat \\
        --captions-root output/caption_gen_unified19 \\
        --events-root output/eval_unified19 \\
        --limit 2 --n-events 1 --output /tmp/judge_smoke
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).parent))

from emotion_query_pipeline import caption_judge as cj
from emotion_query_pipeline.io_utils import read_jsonl, write_jsonl


# ---------------------------------------------------------------------------
# Loading a model's run artefacts
# ---------------------------------------------------------------------------
def _load_captions(captions_root: Path, model: str) -> List[dict]:
    return read_jsonl(captions_root / model / "raw_captions.jsonl")


def _load_segment_clip_map(captions_root: Path, model: str, known_video_ids: Set[str]) -> Dict[Tuple[str, str], str]:
    """Map (video_id, segment_id) -> clip_path.

    ``segments.jsonl`` in these run dirs aggregates ALL videos and has NO
    ``video_id`` column, and ``segment_id`` (s001, s002, ...) resets per
    video, so it alone isn't a safe key. Recover ``video_id`` from
    ``clip_path`` (always ``.../<video_id>/<grid_key>/<file>``) by matching
    the path component against the known video_ids from that model's
    captions — same approach as ``rerun_generation._load_segments_by_video``.
    """
    out: Dict[Tuple[str, str], str] = {}
    for r in read_jsonl(captions_root / model / "segments.jsonl"):
        vid = r.get("video_id")
        clip_path = r.get("clip_path")
        if not vid and clip_path:
            matches = set(Path(clip_path).parts) & known_video_ids
            vid = next(iter(matches)) if len(matches) == 1 else None
        if vid is None or not clip_path:
            continue
        out[(vid, r.get("segment_id"))] = clip_path
    return out


def _load_events(events_root: Path, model: str) -> List[dict]:
    return read_jsonl(events_root / model / "emotion_events.jsonl")


def _load_event_segment_clip_map(events_root: Path, model: str, known_video_ids: Set[str]) -> Dict[Tuple[str, str], str]:
    out: Dict[Tuple[str, str], str] = {}
    for r in read_jsonl(events_root / model / "segments.jsonl"):
        vid = r.get("video_id")
        clip_path = r.get("clip_path")
        if not vid and clip_path:
            matches = set(Path(clip_path).parts) & known_video_ids
            vid = next(iter(matches)) if len(matches) == 1 else None
        if vid is None or not clip_path:
            continue
        out[(vid, r.get("segment_id"))] = clip_path
    return out


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------
def _done_ids(path: Path, id_field: str) -> Set[Tuple[str, str]]:
    return {(r.get("video_id"), r.get(id_field)) for r in read_jsonl(path)}


# ---------------------------------------------------------------------------
# Judging one item
# ---------------------------------------------------------------------------
def _judge_caption(
    client, uploader, caption: dict, clip_path: Optional[str], judge_schema: str,
) -> dict:
    video_id = caption.get("video_id")
    segment_id = caption.get("segment_id")
    record: Dict[str, Any] = {"video_id": video_id, "segment_id": segment_id}
    if not clip_path or not Path(clip_path).is_file():
        record["parse_error"] = f"no clip found for segment {segment_id}"
        for dim in cj.CAPTION_SCORE_DIMENSIONS:
            record[dim] = {"score": None, "reason": ""}
        return record
    uploaded = None
    try:
        prompt = cj.build_caption_judge_payload(caption)
        uploaded = uploader.upload(clip_path)
        raw = client.generate_json(prompt, judge_schema, video_uri=uploaded.uri)
        verdict = cj.parse_caption_verdict(raw)
        record.update(verdict)
    except Exception as e:  # judge call/parse failure never aborts the run
        record["parse_error"] = f"judge call failed: {e}"
        for dim in cj.CAPTION_SCORE_DIMENSIONS:
            record.setdefault(dim, {"score": None, "reason": ""})
    finally:
        if uploaded is not None:
            uploader.delete(uploaded)
    return record


def _judge_event(
    client, uploader, event: dict, clip_paths: List[str], judge_schema: str,
) -> dict:
    video_id = event.get("video_id")
    event_id = event.get("event_id")
    assigned_label = event.get("emotion_label", "")
    record: Dict[str, Any] = {
        "video_id": video_id, "event_id": event_id, "emotion_label": assigned_label,
    }
    valid_clips = [p for p in clip_paths if p and Path(p).is_file()]
    if not valid_clips:
        record["parse_error"] = f"no clip(s) found for event {event_id}"
        for dim in cj.EVENT_SCORE_DIMENSIONS:
            record[dim] = {"score": None, "reason": ""}
        return record
    uploaded_files = []
    try:
        prompt = cj.build_event_judge_payload(event)
        uris = []
        for p in valid_clips:
            f = uploader.upload(p)
            uploaded_files.append(f)
            uris.append(f.uri)
        raw = client.generate_json(prompt, judge_schema, video_uri=uris)
        verdict = cj.parse_event_verdict(raw, assigned_label)
        record.update(verdict)
    except Exception as e:
        record["parse_error"] = f"judge call failed: {e}"
        for dim in cj.EVENT_SCORE_DIMENSIONS:
            record.setdefault(dim, {"score": None, "reason": ""})
    finally:
        for f in uploaded_files:
            uploader.delete(f)
    return record


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Reference-free MLLM-as-judge for caption + emotion-event quality."
    )
    p.add_argument("--models", required=True,
                    help="Comma-separated model subdir names, e.g. nemotron_omni,qwen3_omni,timechat")
    p.add_argument("--captions-root", required=True,
                    help="Dir containing <model>/raw_captions.jsonl + segments.jsonl")
    p.add_argument("--events-root", required=True,
                    help="Dir containing <model>/emotion_events.jsonl + segments.jsonl")
    p.add_argument("--video-dir", default=None,
                    help="Unused for scoring (clips are already cut); kept for parity with other CLIs.")
    p.add_argument("--n-captions", type=int, default=100,
                    help="Captions to sample per model, spread across videos.")
    p.add_argument("--n-events", type=int, default=50,
                    help="Emotion events to sample per model, spread across videos.")
    p.add_argument("--judge-model", default="gemini-2.5-flash",
                    help="Multimodal Gemini judge model (do NOT use a -lite tier).")
    p.add_argument("--output", required=True, help="Output directory for verdicts + aggregate.")
    p.add_argument("--workers", type=int, default=4,
                   help="Concurrent judge calls (ThreadPool). Each call is an independent "
                        "clip-upload + Gemini generate; 4 is safe for pro rate limits.")
    p.add_argument("--limit", type=int, default=0,
                    help="Cap items judged PER MODEL (0 = judge the full sample). For smoke tests.")
    p.add_argument("--api-key", default=None, help="Defaults to GEMINI_API_KEY env var.")
    return p


_CAPTION_JUDGE_SCHEMA = "CaptionQualityJudgeOutput"
_EVENT_JUDGE_SCHEMA = "EmotionEventJudgeOutput"


def _judge_concurrent(pending, out_path, workers, label, judge_fn):
    """Judge `pending` items concurrently (ThreadPool); append each verdict as it
    completes. Each judge_fn call is an independent clip-upload + Gemini generate,
    so this only parallelizes I/O-bound API waits. Writes are lock-guarded."""
    import concurrent.futures as _cf
    import threading as _th
    total = len(pending)
    if total == 0:
        print(f"  {label}: nothing to do (all scored)")
        return
    lock = _th.Lock()
    n = [0]
    with out_path.open("a", encoding="utf-8") as f, \
         _cf.ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = [ex.submit(judge_fn, it) for it in pending]
        for fut in _cf.as_completed(futs):
            try:
                rec = fut.result()
            except Exception as e:  # never let one item kill the run
                rec = {"parse_error": f"judge_fn raised: {e}"}
            with lock:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                n[0] += 1
                status = "ERR" if rec.get("parse_error") else "ok"
                print(f"  [{n[0]}/{total}] {label} -> {status}", flush=True)


def main() -> None:
    args = _build_arg_parser().parse_args()

    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    from emotion_query_pipeline.captioning import GeminiUploader
    from emotion_query_pipeline.llm_client import GeminiLLMClient

    # Every schema_name not in GeminiLLMClient._STAGE falls back to
    # generation_model, so pointing ALL model slots at --judge-model makes any
    # schema_name resolve to the judge model without touching llm_client.py.
    client = GeminiLLMClient(
        caption_model=args.judge_model,
        generation_model=args.judge_model,
        verification_model=args.judge_model,
        rewrite_model=args.judge_model,
        emotion_event_model=args.judge_model,
        api_key=api_key,
    )
    uploader = GeminiUploader(api_key=api_key)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    captions_root = Path(args.captions_root)
    events_root = Path(args.events_root)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    aggregate_report: Dict[str, Any] = {}
    run_start = time.perf_counter()

    for model in models:
        print(f"\n=== {model} ===")
        model_dir = output_dir / model
        model_dir.mkdir(parents=True, exist_ok=True)
        caption_path = model_dir / "caption_verdicts.jsonl"
        event_path = model_dir / "event_verdicts.jsonl"

        # ---------------- captions ----------------
        captions = _load_captions(captions_root, model)
        known_video_ids = {c["video_id"] for c in captions if c.get("video_id")}
        clip_map = _load_segment_clip_map(captions_root, model, known_video_ids)
        n_cap = args.n_captions if args.limit <= 0 else min(args.n_captions, args.limit)
        sampled_captions = cj.sample_segments(captions, n_cap)
        done = _done_ids(caption_path, "segment_id")
        print(f"  captions: sampled {len(sampled_captions)} "
              f"({len(done)} already scored, resuming)")

        pending_caps = [c for c in sampled_captions
                        if (c.get("video_id"), c.get("segment_id")) not in done]
        _judge_concurrent(
            pending_caps, caption_path, args.workers, "caption",
            lambda cap: _judge_caption(
                client, uploader, cap,
                clip_map.get((cap.get("video_id"), cap.get("segment_id"))),
                _CAPTION_JUDGE_SCHEMA),
        )

        # ---------------- emotion events ----------------
        events = _load_events(events_root, model)
        ev_known_video_ids = {e["video_id"] for e in events if e.get("video_id")}
        ev_clip_map = _load_event_segment_clip_map(events_root, model, ev_known_video_ids)
        n_ev = args.n_events if args.limit <= 0 else min(args.n_events, args.limit)
        sampled_events = cj.sample_events(events, n_ev)
        ev_done = _done_ids(event_path, "event_id")
        print(f"  events: sampled {len(sampled_events)} "
              f"({len(ev_done)} already scored, resuming)")

        def _ev_clips(ev):
            return [ev_clip_map[(ev.get("video_id"), sid)]
                    for sid in (ev.get("segment_ids") or [])
                    if (ev.get("video_id"), sid) in ev_clip_map]
        pending_evs = [e for e in sampled_events
                       if (e.get("video_id"), e.get("event_id")) not in ev_done]
        _judge_concurrent(
            pending_evs, event_path, args.workers, "event",
            lambda ev: _judge_event(client, uploader, ev, _ev_clips(ev), _EVENT_JUDGE_SCHEMA),
        )

        # ---------------- aggregate for this model ----------------
        cap_verdicts = read_jsonl(caption_path)
        ev_verdicts = read_jsonl(event_path)
        cap_agg = cj.aggregate(cap_verdicts, cj.CAPTION_SCORE_DIMENSIONS)
        ev_agg = cj.aggregate(ev_verdicts, cj.EVENT_SCORE_DIMENSIONS)
        aggregate_report[model] = {
            "caption": {**cap_agg, "combined_score": cj.combined_caption_score(cap_agg)},
            "event": ev_agg,
        }
        print(f"  caption aggregate: {aggregate_report[model]['caption']}")
        print(f"  event aggregate:   {aggregate_report[model]['event']}")

    with (output_dir / "aggregate.json").open("w", encoding="utf-8") as f:
        json.dump(aggregate_report, f, indent=2, ensure_ascii=False)

    print(f"\nDone in {time.perf_counter() - run_start:.0f}s. "
          f"Aggregate written to {output_dir / 'aggregate.json'}")


if __name__ == "__main__":
    main()
