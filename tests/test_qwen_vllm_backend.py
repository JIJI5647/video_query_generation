"""Tests for the vLLM-served Qwen3-Omni verify backend (``qwen_omni_vllm``).

Pure-Python / offline: no GPU, no network. ``NemotronOpenAIClient`` (the shared
OpenAI-compatible HTTP shim used by both the ``nemotron`` and ``qwen_omni_vllm``
verify backends) is exercised directly, monkeypatching ``_post_once`` so no real
HTTP call is made. Covers:

* payload building — qwen (Instruct, enable_thinking=None) omits
  ``chat_template_kwargs`` entirely; nemotron keeps sending ``enable_thinking``
  (no regression).
* ``generate_json`` / ``generate_json_many`` still parse a canned response
  correctly through the client either way.
* ``run_verification.py`` / ``rerun_generation.py`` argparse wiring: the
  ``qwen_omni_vllm`` choice is accepted and the new ``--qwen-vllm-*`` flags parse.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from emotion_query_pipeline.nemotron_client import NemotronOpenAIClient

REPO_ROOT = Path(__file__).parent.parent


def test_qwen_vllm_payload_omits_enable_thinking():
    """enable_thinking=None (Instruct default) -> no chat_template_kwargs at all."""
    client = NemotronOpenAIClient(
        base_url="http://0.0.0.0:8000/v1",
        model="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        enable_thinking=None,
    )
    payload = client._payload("describe the clip", "/tmp/fake_clip.mp4")
    assert "chat_template_kwargs" not in payload["extra_body"]
    # mm_processor_kwargs is still sent (Qwen3-Omni supports use_audio_in_video).
    assert payload["extra_body"]["mm_processor_kwargs"] == {"use_audio_in_video": False}
    assert payload["model"] == "Qwen/Qwen3-Omni-30B-A3B-Instruct"


def test_qwen_vllm_thinking_checkpoint_sends_enable_thinking_true():
    """--qwen-vllm-thinking maps to enable_thinking=True (Thinking checkpoint)."""
    client = NemotronOpenAIClient(
        base_url="http://0.0.0.0:8000/v1",
        model="Qwen/Qwen3-Omni-30B-A3B-Thinking",
        enable_thinking=True,
    )
    payload = client._payload("describe the clip", None)
    assert payload["extra_body"]["chat_template_kwargs"] == {"enable_thinking": True}


def test_nemotron_payload_still_includes_enable_thinking_no_regression():
    """The existing nemotron backend must be unaffected by the qwen change."""
    client = NemotronOpenAIClient(
        base_url="http://0.0.0.0:8000/v1",
        model="nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8",
        # default enable_thinking=True, same as run_verification.py's default
        # (not args.nemotron_no_thinking).
    )
    payload = client._payload("verify this query", "/tmp/fake_clip.mp4")
    assert payload["extra_body"]["chat_template_kwargs"] == {"enable_thinking": True}
    assert payload["model"] == "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8"

    # --nemotron-no-thinking -> enable_thinking=False (kwarg still sent, just False).
    client_no_think = NemotronOpenAIClient(
        base_url="http://0.0.0.0:8000/v1",
        model="nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8",
        enable_thinking=False,
    )
    payload_no_think = client_no_think._payload("verify this query", None)
    assert payload_no_think["extra_body"]["chat_template_kwargs"] == {
        "enable_thinking": False
    }


def _canned_response(client, payload):
    return {"decision": "pass", "failure_reason": None}


def test_generate_json_parses_through_qwen_client(monkeypatch):
    client = NemotronOpenAIClient(
        base_url="http://0.0.0.0:8000/v1",
        model="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        enable_thinking=None,
    )
    monkeypatch.setattr(client, "_post_once", lambda payload: _canned_response(client, payload))
    result = client.generate_json("verify this query", "verification", video_uri=None)
    assert result == {"decision": "pass", "failure_reason": None}


def test_generate_json_many_preserves_order(monkeypatch):
    client = NemotronOpenAIClient(
        base_url="http://0.0.0.0:8000/v1",
        model="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        enable_thinking=None,
        max_workers=2,
    )

    def fake_post_once(payload):
        # Echo back which prompt this payload was for, to check ordering.
        text = payload["messages"][0]["content"]
        return {"echo": text}

    monkeypatch.setattr(client, "_post_once", fake_post_once)
    prompts = ["prompt A", "prompt B", "prompt C"]
    results = client.generate_json_many(prompts, "verification")
    assert [r["echo"] for r in results] == prompts


def test_generate_json_many_isolates_failures(monkeypatch):
    """One query failing all retries returns {} without losing the rest of the chunk."""
    client = NemotronOpenAIClient(
        base_url="http://0.0.0.0:8000/v1",
        model="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        enable_thinking=None,
        max_retries=1,
        max_workers=1,
    )

    def flaky_post_once(payload):
        text = payload["messages"][0]["content"]
        if text == "bad prompt":
            raise ValueError("boom")
        return {"decision": "pass"}

    monkeypatch.setattr(client, "_post_once", flaky_post_once)
    results = client.generate_json_many(["ok prompt", "bad prompt"], "verification")
    assert results == [{"decision": "pass"}, {}]


def test_run_verification_help_lists_qwen_omni_vllm_backend():
    out = subprocess.run(
        [sys.executable, str(REPO_ROOT / "run_verification.py"), "--help"],
        capture_output=True, text=True, check=True,
    )
    assert "qwen_omni_vllm" in out.stdout
    assert "--qwen-vllm-base-url" in out.stdout
    assert "--qwen-vllm-model" in out.stdout
    assert "--qwen-vllm-max-tokens" in out.stdout
    assert "--qwen-vllm-thinking" in out.stdout
    # Existing nemotron flags must still be present (no regression).
    assert "--nemotron-base-url" in out.stdout


def test_rerun_generation_help_lists_qwen_omni_vllm_backend():
    out = subprocess.run(
        [sys.executable, str(REPO_ROOT / "rerun_generation.py"), "--help"],
        capture_output=True, text=True, check=True,
    )
    assert "qwen_omni_vllm" in out.stdout
    assert "--qwen-vllm-base-url" in out.stdout
    assert "--nemotron-base-url" in out.stdout
