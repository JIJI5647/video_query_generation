"""Step 2+3: batch clips -> upload -> emotion captions.

Segments are grouped into consecutive batches of ``batch_size``. Each batch
uploads its N clips to the Gemini Files API and issues ONE multimodal call
(N video parts + one text prompt). The prompt enumerates the clip -> segment_id
order explicitly and asks for exactly one caption per clip. We emit exactly one
caption per segment (1-to-1 with the clip), keyed by the segment_id the model
reports; any segment the model skips or mislabels gets a neutral placeholder.
All captions (including neutral / "unrelevant") are fed to generation, which
selects which moments are worth a query.

``GeminiUploader`` is a thin wrapper over the Files API (upload + poll ACTIVE +
delete). It is reused by the entry script to upload the whole video for the
verify/rewrite stage. The LLM boundary stays behind ``BaseLLMClient`` so the
pure logic here is testable with a fake uploader + fake client.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import List, Optional

from .io_utils import load_prompt_template
from .llm_client import BaseLLMClient
from .models import CaptionBatchOutput, EmotionCaption, Segment

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


# ---------------------------------------------------------------------------
# Files API uploader
# ---------------------------------------------------------------------------
class GeminiUploader:
    """Upload local files to the Gemini Files API and poll until ACTIVE."""

    _UPLOAD_TIMEOUT = 180.0
    _UPLOAD_POLL_INTERVAL = 2.0

    def __init__(self, api_key: Optional[str] = None) -> None:
        from google import genai  # lazy import: keep SDK out of pure modules

        resolved_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "No Gemini API key provided. "
                "Pass api_key= or set the GEMINI_API_KEY environment variable."
            )
        self._client = genai.Client(api_key=resolved_key)

    def upload(self, file_path: str):
        file_obj = self._client.files.upload(file=file_path)
        deadline = time.time() + self._UPLOAD_TIMEOUT
        while file_obj.state.name != "ACTIVE":
            if file_obj.state.name == "FAILED":
                raise RuntimeError(f"Upload failed for {file_path}")
            if time.time() > deadline:
                raise TimeoutError(
                    f"{file_path} not ACTIVE after {self._UPLOAD_TIMEOUT}s"
                )
            time.sleep(self._UPLOAD_POLL_INTERVAL)
            file_obj = self._client.files.get(name=file_obj.name)
        return file_obj

    def delete(self, file_obj) -> None:
        try:
            self._client.files.delete(name=file_obj.name)
        except Exception:  # best-effort cleanup
            pass


# ---------------------------------------------------------------------------
# Batch captioning
# ---------------------------------------------------------------------------
def _batches(segments: List[Segment], batch_size: int) -> List[List[Segment]]:
    return [
        segments[i : i + batch_size]
        for i in range(0, len(segments), batch_size)
    ]


def build_caption_prompt(
    video_id: str,
    batch_index: int,
    batch: List[Segment],
    prompts_dir: Optional[Path] = None,
) -> str:
    template = load_prompt_template(
        prompts_dir or _PROMPTS_DIR, "caption_prompt.txt"
    )
    lines = []
    for clip_no, seg in enumerate(batch, 1):
        lines.append(
            f"Clip {clip_no} -> {seg.segment_id} "
            f"({seg.start_time:.2f}-{seg.end_time:.2f}s)"
        )
    prompt = template
    prompt = prompt.replace("{video_id}", video_id)
    prompt = prompt.replace("{batch_index}", str(batch_index))
    prompt = prompt.replace("{segment_list}", "\n".join(lines))
    return prompt


def _neutral_placeholder(video_id: str, seg_id: str) -> EmotionCaption:
    """A 'no emotion observed' caption for a segment the model skipped.

    Kept so captions are exactly one-per-segment; generation decides whether a
    neutral moment is worth a query.
    """
    return EmotionCaption(
        video_id=video_id,
        caption_id=f"{video_id}_{seg_id}",
        segment_ids=[seg_id],
        person="not described",
        action="not described",
        sound="no audible cue",
        emotion="neutral",
        confidence="low",
        evidence_strength="ambiguous",
        observable_evidence=[],
    )


def caption_batch(
    video_id: str,
    batch_index: int,
    batch: List[Segment],
    video_uris: List[str],
    client: BaseLLMClient,
    prompts_dir: Optional[Path] = None,
) -> CaptionBatchOutput:
    """Caption one batch of contiguous segments (one multimodal LLM call).

    Produces exactly one caption per segment in the batch (1-to-1 with the
    clip). The model is asked for one caption per clip; we key its output by the
    segment_id it reports and emit captions in segment order, synthesizing a
    neutral placeholder for any segment the model skipped or mislabelled.
    """
    valid_ids = {seg.segment_id for seg in batch}
    prompt = build_caption_prompt(video_id, batch_index, batch, prompts_dir)
    raw = client.generate_json(prompt, "CaptionBatchOutput", video_uri=video_uris)

    # Index the model's captions by the (first valid) segment_id they claim.
    by_seg: dict[str, dict] = {}
    for item in raw.get("captions") or []:
        seg_ids = [s for s in (item.get("segment_ids") or []) if s in valid_ids]
        if not seg_ids:
            continue
        seg_id = seg_ids[0]
        by_seg.setdefault(seg_id, item)  # first caption wins for a segment

    captions: List[EmotionCaption] = []
    for seg in batch:
        seg_id = seg.segment_id
        item = by_seg.get(seg_id)
        if item is None:
            captions.append(_neutral_placeholder(video_id, seg_id))
            continue
        item["segment_ids"] = [seg_id]  # force the one segment of this clip
        item["video_id"] = video_id
        item["caption_id"] = f"{video_id}_{seg_id}"  # 1-to-1 with the segment
        try:
            captions.append(EmotionCaption.model_validate(item))
        except Exception as e:  # fall back to placeholder rather than abort batch
            print(f"    [caption fallback] {seg_id}: {e}")
            captions.append(_neutral_placeholder(video_id, seg_id))
    return CaptionBatchOutput(
        video_id=video_id,
        batch_index=batch_index,
        segment_ids=sorted(valid_ids),
        captions=captions,
    )


def caption_video(
    video_id: str,
    segments: List[Segment],
    client: BaseLLMClient,
    uploader: GeminiUploader,
    batch_size: int = 8,
    prompts_dir: Optional[Path] = None,
) -> List[EmotionCaption]:
    """Caption every segment of a video, batch by batch. Returns raw captions.

    Uploads each batch's clips, calls the model once, then deletes the uploaded
    clip references for that batch before moving on.
    """
    raw_captions: List[EmotionCaption] = []
    for batch_index, batch in enumerate(_batches(segments, batch_size), 1):
        usable = [seg for seg in batch if seg.clip_path]
        if not usable:
            continue
        uploaded = []
        try:
            for seg in usable:
                uploaded.append(uploader.upload(seg.clip_path))
            uris = [f.uri for f in uploaded]
            batch_out = caption_batch(
                video_id, batch_index, usable, uris, client, prompts_dir
            )
            raw_captions.extend(batch_out.captions)
        finally:
            for f in uploaded:
                uploader.delete(f)
    return raw_captions
