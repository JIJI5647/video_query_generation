"""LLM client for an Omni model served over an OpenAI-compatible endpoint.

The verify/rewrite stages can run on a served omni model instead of the in-process
Qwen3-Omni ``transformers`` engine. This client is model-agnostic — it just needs a
``base_url`` + ``model`` id — so it is reused for two served backends:

* NVIDIA's Nemotron-3-Nano-Omni reasoning model (``--verify-rewrite-backend
  nemotron``).
* A vLLM-served Qwen3-Omni (``--verify-rewrite-backend qwen_omni_vllm``), for
  running qwen verification over an HTTP server (continuous batching) instead of
  in-process ``transformers``. The Qwen3-Omni **Instruct** checkpoint has no
  thinking mode, so callers targeting it should pass ``enable_thinking=None`` to
  suppress the ``chat_template_kwargs`` key entirely (the Thinking checkpoint can
  still pass ``enable_thinking=True``).

The class name stays ``NemotronOpenAIClient`` for backward compatibility with
existing call sites (``run_nemotron_sweep.sh`` etc.); it is not Nemotron-specific.

Unlike the Qwen backend (in-process ``transformers``), served models are HTTP
servers exposing the OpenAI ``/v1/chat/completions`` API, so the client is a thin
HTTP shim. The SAME code targets both NVIDIA serving stacks for Nemotron because
they expose the identical OpenAI-compatible API:

* ``trtllm-serve`` (TensorRT-LLM) -- the efficient NVIDIA inference framework this
  backend is designed for. Requires TensorRT-LLM 1.3.0rc+ which is built against
  CUDA 13 (base driver >= R580).
* ``vllm serve`` -- same API; usable on CUDA 12 / older drivers when the installed
  vLLM has the Nemotron-omni architecture registered.

Because the transport is identical, swapping the serving engine is a one-line change
to the *server launch command* only -- this client is unchanged. It is duck-typed to
``BaseLLMClient`` (implements ``generate_json`` / ``generate_json_many``) and, like
``QwenOmniLLMClient``, takes LOCAL clip path(s) as ``video_uri`` (no upload) which it
converts to ``file://`` URIs. The server must therefore be started with local-media
access enabled (e.g. ``vllm serve ... --allowed-local-media-path /``).
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

VideoUriArg = Optional[Union[str, List[str]]]


def _extract_json(text: str) -> Dict[str, Any]:
    """Pull the first JSON object out of a model message, tolerating prose/fences.

    Reasoning models may wrap the answer in ``<think>...</think>`` or a ```json
    fence, or emit chain-of-thought before the JSON. Mirrors the robustness of the
    Gemini client: strip a fence, drop a leading think-block, then ``raw_decode``
    from the first ``{`` (ignoring trailing junk).
    """
    if not text:
        raise ValueError("empty response text")
    t = text.strip()
    # Drop a leading <think>...</think> reasoning block if the server folded the
    # reasoning back into content instead of a separate reasoning_content field.
    lower = t.lower()
    if "</think>" in lower:
        t = t[lower.rindex("</think>") + len("</think>"):].strip()
    if "```" in t:
        fence = t.find("```")
        rest = t[fence + 3:]
        if "\n" in rest:
            rest = rest.split("\n", 1)[1]
        end = rest.rfind("```")
        t = (rest[:end] if end != -1 else rest).strip()
    start = t.find("{")
    if start == -1:
        raise ValueError("no JSON object found in response")
    return json.JSONDecoder().raw_decode(t[start:])[0]


class NemotronOpenAIClient:
    """``BaseLLMClient``-compatible client hitting an OpenAI-compatible server.

    Args:
        base_url: e.g. ``http://0.0.0.0:8000/v1``.
        model: served model id (the HF repo id passed to trtllm-serve / vllm serve).
        max_tokens: generation budget. A reasoning model needs a large budget so the
            chain-of-thought does not truncate before the JSON (analogous to the
            Qwen Thinking checkpoint's ``--qwen-max-tokens 8192``).
        enable_thinking: toggles the model's reasoning trace via chat_template_kwargs.
            ``True``/``False`` send the kwarg explicitly (Nemotron always wants one
            of these); ``None`` omits ``chat_template_kwargs`` entirely, for servers
            (e.g. Qwen3-Omni Instruct) that don't support/expect it.
        use_audio_in_video: passed through as an mm_processor kwarg; verify prompts
            judge visual answerability, audio off by default (matches the model
            card's video example).
        max_workers: how many requests in a chunk are fired concurrently (the HTTP
            analog of the Qwen engine's single batched forward; the server's
            continuous batching does the real batching).
    """

    def __init__(
        self,
        base_url: str = "http://0.0.0.0:8000/v1",
        model: str = "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8",
        max_tokens: int = 8192,
        temperature: float = 0.6,
        top_p: float = 0.95,
        enable_thinking: Optional[bool] = True,
        use_audio_in_video: bool = False,
        max_retries: int = 3,
        timeout: float = 900.0,
        max_workers: int = 4,
        api_key: str = "null",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.enable_thinking = enable_thinking
        self.use_audio_in_video = use_audio_in_video
        self.max_retries = max(1, max_retries)
        self.timeout = timeout
        self.max_workers = max(1, max_workers)
        self.api_key = api_key

    # -- request building ----------------------------------------------------
    @staticmethod
    def _clip_uris(video_uri: VideoUriArg) -> List[str]:
        """Local clip path(s) -> absolute ``file://`` URI(s)."""
        if video_uri is None:
            return []
        paths = [video_uri] if isinstance(video_uri, str) else list(video_uri)
        return [Path(p).resolve().as_uri() for p in paths if p]

    def _build_messages(self, prompt: str, video_uri: VideoUriArg) -> list:
        uris = self._clip_uris(video_uri)
        if not uris:
            return [{"role": "user", "content": prompt}]
        content: List[Dict[str, Any]] = [
            {"type": "video_url", "video_url": {"url": u}} for u in uris
        ]
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    def _payload(self, prompt: str, video_uri: VideoUriArg) -> Dict[str, Any]:
        extra_body: Dict[str, Any] = {
            "mm_processor_kwargs": {"use_audio_in_video": self.use_audio_in_video},
        }
        # Omit chat_template_kwargs entirely when enable_thinking is None (e.g. the
        # Qwen3-Omni Instruct checkpoint, which has no thinking mode) rather than
        # sending an unsupported/ignored key.
        if self.enable_thinking is not None:
            extra_body["chat_template_kwargs"] = {
                "enable_thinking": self.enable_thinking
            }
        return {
            "model": self.model,
            "messages": self._build_messages(prompt, video_uri),
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "extra_body": extra_body,
        }

    def _post_once(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = resp.read().decode("utf-8")
        obj = json.loads(body)
        msg = obj["choices"][0]["message"]
        # Prefer the parsed answer (content); fall back to reasoning_content only if
        # content is empty (e.g. budget exhausted mid-think).
        text = msg.get("content") or msg.get("reasoning_content") or ""
        return _extract_json(text)

    # -- BaseLLMClient interface --------------------------------------------
    def generate_json(
        self, prompt: str, schema_name: str, video_uri: VideoUriArg = None
    ) -> Dict[str, Any]:
        payload = self._payload(prompt, video_uri)
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return self._post_once(payload)
            except (urllib.error.URLError, urllib.error.HTTPError) as e:
                last_error = e
                detail = ""
                if isinstance(e, urllib.error.HTTPError):
                    try:
                        detail = e.read().decode("utf-8")[:300]
                    except Exception:
                        pass
                print(f"  [attempt {attempt}/{self.max_retries}] Nemotron HTTP "
                      f"error for {schema_name}: {e} {detail}. Retrying...")
            except (ValueError, KeyError, json.JSONDecodeError) as e:
                last_error = e
                print(f"  [attempt {attempt}/{self.max_retries}] Nemotron parse "
                      f"error for {schema_name}: {e}. Retrying...")
            if attempt < self.max_retries:
                time.sleep(min(5.0 * (2 ** (attempt - 1)), 30.0))
        raise RuntimeError(
            f"Nemotron call failed after {self.max_retries} attempts "
            f"(schema={schema_name}): {last_error}"
        )

    def generate_json_many(
        self,
        prompts: List[str],
        schema_name: str,
        video_uris: Optional[List[VideoUriArg]] = None,
    ) -> List[Dict[str, Any]]:
        """Fire a chunk of requests concurrently; order preserved.

        The server (trtllm-serve / vllm) batches concurrent requests internally, so
        this is the HTTP analog of the Qwen engine's single batched forward. A query
        that fails all retries returns ``{}`` (which the per-dimension verifier reads
        as a fail-safe "invalid format") rather than raising, so one bad query never
        loses the rest of the chunk -- same blast-radius guarantee as the Qwen path.
        """
        if not prompts:
            return []
        uris = list(video_uris) if video_uris else [None] * len(prompts)

        def _one(pair):
            p, u = pair
            try:
                return self.generate_json(p, schema_name, video_uri=u)
            except Exception as e:
                print(f"  WARN: Nemotron query failed, marking invalid: {e}")
                return {}

        workers = min(self.max_workers, len(prompts))
        if workers == 1:
            return [_one(pair) for pair in zip(prompts, uris)]
        with ThreadPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(_one, zip(prompts, uris)))

    def usage_report(self) -> Dict[str, Any]:
        """Interface parity with GeminiLLMClient (served model: no token counts)."""
        empty = {
            "calls": 0, "prompt_tokens": 0,
            "candidates_tokens": 0, "total_tokens": 0,
        }
        return {"by_stage": {}, "total": dict(empty)}
