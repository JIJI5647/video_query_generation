"""LLM client interface and Gemini implementation.

v2 differences from v1:
- ``generate_json`` accepts ``video_uri`` as ``None`` (caption-only / text call),
  a single str (one whole-video part, used by verify/rewrite), OR a list of
  str (N clip parts in one call, used by batch captioning).
- A dedicated ``caption_model`` is selected for ``"CaptionBatchOutput"`` calls.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional, Union

# NOTE: ``google.genai`` is imported lazily (inside the methods that use it) so
# that ``BaseLLMClient`` and the pure helpers in this module can be imported
# without the SDK installed (e.g. for local tests of generation / workflow).

VideoUriArg = Optional[Union[str, List[str]]]


class BaseLLMClient:
    """Abstract interface for LLM clients used in the pipeline."""

    def generate_json(
        self,
        prompt: str,
        schema_name: str,
        video_uri: VideoUriArg = None,
    ) -> Dict[str, Any]:
        """Send a prompt (and optionally video parts) and return parsed JSON.

        Args:
            prompt: Text prompt to send to the model.
            schema_name: One of "CaptionBatchOutput", "GenerationOutput",
                "VerificationBatchOutput", "RewriteBatchOutput". Selects the model.
            video_uri: ``None`` for a text-only call; a single Files API URI for
                one video part; or a list of URIs for multiple clip parts. Video
                parts are placed before the text prompt so the model watches
                first.
        """
        raise NotImplementedError

    def generate_json_many(
        self,
        prompts: List[str],
        schema_name: str,
        video_uris: Optional[List[VideoUriArg]] = None,
    ) -> List[Dict[str, Any]]:
        """Run N independent ``generate_json`` requests, output order preserved.

        Each ``prompts[i]`` is paired with ``video_uris[i]`` (its own clip(s)).
        The base implementation is sequential; backends that can truly batch
        (e.g. the local Qwen3-Omni engine, one forward over N prompts) override
        this for throughput — the analog of caption batching.
        """
        uris = video_uris or [None] * len(prompts)
        return [
            self.generate_json(p, schema_name, video_uri=u)
            for p, u in zip(prompts, uris)
        ]


class GeminiLLMClient(BaseLLMClient):
    """Production client backed by the Gemini API via the google-genai SDK."""

    def __init__(
        self,
        caption_model: str = "gemini-2.5-flash-lite",
        generation_model: str = "gemini-2.5-flash-lite",
        verification_model: str = "gemini-3.1-flash-lite",
        rewrite_model: str = "gemini-2.5-flash-lite",
        emotion_event_model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 1.0,
        max_retries: int = 5,
        retry_delay: float = 5.0,
    ) -> None:
        self.caption_model = caption_model
        self.generation_model = generation_model
        self.verification_model = verification_model
        self.rewrite_model = rewrite_model
        # Emotion-event stage defaults to the generation model unless overridden.
        self.emotion_event_model = emotion_event_model or generation_model
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        from google import genai  # lazy: keep the SDK off the module import path

        resolved_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "No Gemini API key provided. "
                "Pass api_key= or set the GEMINI_API_KEY environment variable."
            )
        self._client = genai.Client(api_key=resolved_key)
        # Cumulative token usage, keyed by schema_name.
        self._usage: Dict[str, Dict[str, int]] = {}

    # Map schema names to human-readable pipeline stages.
    _STAGE = {
        "CaptionBatchOutput": "caption",
        "EmotionEventOutput": "emotion_event",
        "GenerationOutput": "generation",
        "RegroundingOutput": "reground",
        "VerificationBatchOutput": "verification",
        "RewriteBatchOutput": "rewrite",
    }

    def _record_usage(self, schema_name: str, response: Any) -> None:
        meta = getattr(response, "usage_metadata", None)
        bucket = self._usage.setdefault(
            schema_name,
            {"calls": 0, "prompt_tokens": 0, "candidates_tokens": 0, "total_tokens": 0},
        )
        bucket["calls"] += 1
        if meta is not None:
            bucket["prompt_tokens"] += getattr(meta, "prompt_token_count", 0) or 0
            bucket["candidates_tokens"] += getattr(meta, "candidates_token_count", 0) or 0
            bucket["total_tokens"] += getattr(meta, "total_token_count", 0) or 0

    def usage_report(self) -> Dict[str, Any]:
        """Token usage broken down by stage, plus a grand total."""
        by_stage: Dict[str, Dict[str, int]] = {}
        grand = {"calls": 0, "prompt_tokens": 0, "candidates_tokens": 0, "total_tokens": 0}
        for schema_name, bucket in self._usage.items():
            stage = self._STAGE.get(schema_name, schema_name)
            by_stage[stage] = dict(bucket)
            for k in grand:
                grand[k] += bucket[k]
        return {"by_stage": by_stage, "total": grand}

    def _model_for(self, schema_name: str) -> str:
        if schema_name == "CaptionBatchOutput":
            return self.caption_model
        if schema_name == "EmotionEventOutput":
            return self.emotion_event_model
        if schema_name == "VerificationBatchOutput":
            return self.verification_model
        if schema_name == "RewriteBatchOutput":
            return self.rewrite_model
        return self.generation_model

    @staticmethod
    def _video_parts(video_uri: VideoUriArg) -> List[Any]:
        if video_uri is None:
            return []
        from google.genai import types  # lazy

        uris = [video_uri] if isinstance(video_uri, str) else list(video_uri)
        return [
            types.Part.from_uri(file_uri=uri, mime_type="video/mp4")
            for uri in uris
        ]

    def generate_json(
        self,
        prompt: str,
        schema_name: str,
        video_uri: VideoUriArg = None,
    ) -> Dict[str, Any]:
        from google.genai import types  # lazy

        model = self._model_for(schema_name)
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=self.temperature,
        )

        # Build content parts: video part(s) first (if any), then the text prompt.
        contents: list = self._video_parts(video_uri)
        contents.append(prompt)

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                )
                self._record_usage(schema_name, response)
                raw_text = response.text
                if raw_text is None:
                    raise ValueError(
                        "empty response (no text candidate; possibly safety-blocked)"
                    )
                text = raw_text.strip()
                if "```" in text:
                    # Drop a ```json ... ``` fence if present.
                    fence = text.find("```")
                    rest = text[fence + 3:]
                    if "\n" in rest:
                        rest = rest.split("\n", 1)[1]
                    end = rest.rfind("```")
                    text = (rest[:end] if end != -1 else rest).strip()
                # Skip any leading prose (e.g. chain-of-thought reasoning written
                # before the JSON) by decoding from the first '{'. raw_decode then
                # parses one JSON value and ignores trailing junk.
                start = text.find("{")
                if start == -1:
                    raise ValueError("no JSON object found in response")
                return json.JSONDecoder().raw_decode(text[start:])[0]
            except json.JSONDecodeError as e:
                last_error = e
                print(
                    f"  [attempt {attempt}/{self.max_retries}] JSON parse error "
                    f"for {schema_name}: {e}. Retrying..."
                )
            except Exception as e:
                last_error = e
                print(
                    f"  [attempt {attempt}/{self.max_retries}] API error "
                    f"for {schema_name}: {e}. Retrying..."
                )
            if attempt < self.max_retries:
                # Exponential backoff (5, 10, 20, 40, ... capped at 60s) so a
                # transient 503 "high demand" spike is ridden out, not failed.
                delay = min(self.retry_delay * (2 ** (attempt - 1)), 60.0)
                time.sleep(delay)

        raise RuntimeError(
            f"Gemini call failed after {self.max_retries} attempts "
            f"(schema={schema_name}, model={model}): {last_error}"
        )
