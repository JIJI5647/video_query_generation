"""OPTIONAL server integration test: real caption model + real Gemini downstream.

Skipped unless ``RUN_CAPTION_QUERY_INTEGRATION=1``. On a GPU server with a Gemini
key it runs one caption model on a real clip, feeds the normalized captions
through the REAL Gemini emotion-event + query-generation stages, and asserts the
artefacts are written and validate with the existing Pydantic models.

Env vars:
    RUN_CAPTION_QUERY_INTEGRATION=1   (required to un-skip)
    CAPTION_QUERY_MODEL              (default: qwen3_omni)
    CAPTION_QUERY_VIDEO              (required for AV/video models)
    CAPTION_QUERY_AUDIO             (required for audio+video models)
    CAPTION_QUERY_OUTPUT            (default: output/caption_query_tests/<model>)
    GEMINI_API_KEY                  (required — real downstream)

Run:  RUN_CAPTION_QUERY_INTEGRATION=1 CAPTION_QUERY_VIDEO=short.mp4 \
      python -m pytest tests/test_caption_query_integration.py -q -s
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_CAPTION_QUERY_INTEGRATION") != "1",
    reason="set RUN_CAPTION_QUERY_INTEGRATION=1 to run the server integration test",
)


def test_caption_to_query_integration():
    from emotion_query_pipeline import caption_query_test as cqt
    from emotion_query_pipeline.llm_client import GeminiLLMClient
    from emotion_query_pipeline.models import (
        EmotionEventOutput,
        GenerationOutput,
        OmniCaption,
    )

    model = os.environ.get("CAPTION_QUERY_MODEL", "qwen3_omni")
    video = os.environ.get("CAPTION_QUERY_VIDEO")
    audio = os.environ.get("CAPTION_QUERY_AUDIO")
    output = Path(
        os.environ.get("CAPTION_QUERY_OUTPUT", f"output/caption_query_tests/{model}")
    )
    api_key = os.environ.get("GEMINI_API_KEY")
    assert api_key, "GEMINI_API_KEY required for the integration test"

    spec = cqt.validate_inputs(model, video, audio)

    seg = cqt.make_segment(segment_id="s001", start=0.0, end=5.0, clip_path=video)
    config = cqt.RunnerConfig()
    out = cqt.run_caption_model(
        model, seg, video_path=video, audio_path=audio, config=config
    )
    if out.modality == "audio_video":
        cap = cqt.merge_audio_video_caption(
            out.audio_text, out.video_text, seg, "integration",
            audio_source_model=out.audio_source_model or "",
            video_source_model=out.video_source_model or "",
            source_caption_model=out.source_caption_model,
        )
    else:
        cap = cqt.normalize_to_omni_caption(
            out.raw_output, seg, "integration",
            source_caption_model=out.source_caption_model, modality=out.modality,
        )
    assert isinstance(cap, OmniCaption)
    # Normalized caption must carry SOME observation text.
    assert any([
        cap.audio_description.strip(),
        cap.temporal_description.strip(),
        cap.visual_objective.people,
    ]), "normalized caption is empty"

    # Stage 1 (caption generation) artefacts.
    written = cqt.save_caption_outputs(
        output,
        raw_records=[{"segment_id": "s001", "raw_output": str(out.raw_output)}],
        captions=[cap], segments=[seg],
        metadata={"caption_model": model, "non_commercial": spec.non_commercial},
    )
    assert "segments.jsonl" in written
    assert "run_metadata.json" in written

    # Stage 2 (query generation) artefacts.
    client = GeminiLLMClient(api_key=api_key)
    inputs = cqt.build_downstream_inputs("integration", [cap], [seg])
    downstream = cqt.run_downstream_gemini(inputs, client)
    events, generation = downstream["events"], downstream["generation"]

    written = cqt.save_generation_outputs(
        output, events=events, generation=generation, segments=[seg],
        metadata={
            "caption_model": model,
            "num_generated_queries": len(generation.queries),
            "warnings": downstream["warnings"],
        },
    )

    assert (output / "emotion_events.json").is_file()
    assert (output / "generated_queries.json").is_file()
    # Written artefacts validate with the existing Pydantic models.
    EmotionEventOutput.model_validate(
        json.loads((output / "emotion_events.json").read_text())
    )
    GenerationOutput.model_validate(
        json.loads((output / "generated_queries.json").read_text())
    )
    # Zero queries must be surfaced, not hidden.
    if not generation.queries:
        meta = json.loads((output / "generation_metadata.json").read_text())
        assert meta["warnings"], "0 queries but no warning recorded in metadata"
    assert "generation_metadata.json" in written

    # Round trip: stage 2 can be re-run purely from stage 1's cached output dir.
    reloaded_segments, reloaded_captions = cqt.load_caption_outputs(output)
    assert len(reloaded_segments) == 1
    assert len(reloaded_captions) == 1
