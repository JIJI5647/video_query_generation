"""Efficient batch caption sessions — load each caption model ONCE, reuse it.

The per-segment runners in ``caption_query_test`` (``run_caption_model``) rebuild
the heavy model on every call, which is fine for the single-clip plumbing test but
ruinous for real videos (a 480s clip is ~96 segments; reloading a 7B/30B model per
segment is hours of pure load time). This module wraps the SAME extracted
generate helpers in ``caption_query_test`` in a **session** object that loads the
model(s) once in ``__init__`` and captions each segment via ``.caption(...)``.

``build_caption_session(caption_model, config)`` returns the right session for any
of the six supported models. For the two audio+video pairs whose audio half runs
in a separate conda env (``af3_vl``, ``secap_qwen``) the audio model is kept
loaded across segments via a **persistent subprocess server** (the standalone
runner's ``--server`` mode: load once, then caption one stdin path per line),
rather than one subprocess launch per segment.

IMPORT-SAFE: like ``caption_query_test``, importing this module pulls in NO heavy
deps (torch/transformers/etc are imported lazily inside the extracted generate
helpers, only when a session is actually constructed and used).
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List, Optional

from . import caption_query_test as cqt
from .caption_query_test import CaptionModelOutput, ModelSpec, RunnerConfig
from .models import Segment


# ---------------------------------------------------------------------------
# In-process single-AV-model sessions
# ---------------------------------------------------------------------------
class Qwen3OmniSession:
    """Qwen3-Omni (video + text prompt), loaded once via the pipeline captioner."""

    def __init__(self, spec: ModelSpec, config: RunnerConfig,
                 prompts_dir: Optional[Path] = None) -> None:
        from .omni_captioning import Qwen3OmniCaptioner  # lazy (import-safe)

        self.model_path = config.caption_model_path or spec.default_model_path
        self.prompts_dir = prompts_dir
        # Qwen3OmniCaptioner loads its weights lazily on the first generate() and
        # reuses them for every subsequent call — exactly the reuse we want.
        self._captioner = Qwen3OmniCaptioner(
            model_path=self.model_path,
            attn_implementation=config.attn_impl,
            device_map=config.device_map,
        )

    def caption(self, segment: Segment, video_path: Optional[str],
                audio_path: Optional[str]) -> CaptionModelOutput:
        from .omni_captioning import (  # lazy
            Qwen3OmniCaptioner,
            build_omni_caption_prompt,
        )

        prompt = build_omni_caption_prompt([segment], self.prompts_dir)
        messages = Qwen3OmniCaptioner._build_messages_multi(prompt, [video_path or ""])
        raw = self._captioner.generate(messages)
        return CaptionModelOutput(
            modality="av", raw_output=raw, source_caption_model=self.model_path
        )

    def close(self) -> None:
        pass


class Qwen25OmniAVSession:
    """AVoCaDO / TimeChat (Qwen2.5-Omni-7B fine-tunes), model+processor loaded once."""

    def __init__(self, spec: ModelSpec, config: RunnerConfig, prompt: str,
                 system_prompt: Optional[str] = None, fold_timechat: bool = False) -> None:
        self.spec = spec
        self.config = config
        self.prompt = prompt
        self.system_prompt = system_prompt
        self.fold_timechat = fold_timechat
        self.model_path = config.caption_model_path or spec.default_model_path
        self._model, self._processor = cqt._load_qwen2_5_omni(self.model_path, config)

    def caption(self, segment: Segment, video_path: Optional[str],
                audio_path: Optional[str]) -> CaptionModelOutput:
        raw = cqt._qwen2_5_omni_generate(
            self._model, self._processor, video_path or "", self.prompt,
            self.config, system_prompt=self.system_prompt,
        )
        if self.fold_timechat:
            scenes = cqt._try_parse_json_array(raw)
            if scenes is not None:
                folded = cqt._fold_timechat_scenes(scenes)
                if folded is not None:
                    raw = folded
        return CaptionModelOutput(
            modality="av", raw_output=raw, source_caption_model=self.model_path
        )

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Sub-sessions for the audio+video pairs (video half + audio half)
# ---------------------------------------------------------------------------
class Qwen3VLVideoSession:
    """Qwen3-VL video-only captioner, model+processor loaded once."""

    def __init__(self, model_path: str, config: RunnerConfig) -> None:
        self.model_path = model_path
        self.config = config
        self._model, self._processor = cqt._load_qwen3_vl(model_path, config)

    def caption_video(self, video_path: str) -> str:
        return cqt._qwen3_vl_generate(self._model, self._processor, video_path, self.config)

    def close(self) -> None:
        pass


class QwenOmniCaptionerAudioSession:
    """Qwen3-Omni-Captioner audio-only captioner (no text prompt), loaded once."""

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self._model, self._processor = cqt._load_qwen_omni_captioner(model_path)

    def caption_audio(self, audio_path: str) -> str:
        return cqt._qwen_omni_captioner_generate(self._model, self._processor, audio_path)

    def close(self) -> None:
        pass


class SubprocessAudioSession:
    """Audio captioner kept loaded in a persistent subprocess (``--server`` mode).

    Spawns the standalone runner once (in its own conda env), waits for its
    ``###READY###`` line, then for each ``caption_audio`` writes one absolute
    audio path + newline and reads back the caption between the
    ``###CAPTION_START###`` / ``###CAPTION_END###`` markers. This keeps the audio
    model resident across all segments of a run instead of reloading it per clip.
    """

    def __init__(self, cmd: List[str], model_path: str, cwd: Optional[str] = None,
                 startup_timeout: float = 1200.0) -> None:
        self.model_path = model_path
        self._proc = subprocess.Popen(
            cmd, cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1,
        )
        # Wait for the server to finish loading the model.
        while True:
            line = self._proc.stdout.readline()
            if line == "":
                err = self._proc.stderr.read() if self._proc.stderr else ""
                raise RuntimeError(
                    f"audio server exited before ready: {' '.join(cmd)}\n"
                    f"--- stderr (tail) ---\n{err[-4000:]}"
                )
            if line.strip() == "###READY###":
                break

    def caption_audio(self, audio_path: str) -> str:
        if self._proc.poll() is not None:
            err = self._proc.stderr.read() if self._proc.stderr else ""
            raise RuntimeError(
                f"audio server died (exit {self._proc.returncode}).\n"
                f"--- stderr (tail) ---\n{err[-4000:]}"
            )
        self._proc.stdin.write(str(Path(audio_path).resolve()) + "\n")
        self._proc.stdin.flush()

        lines: List[str] = []
        started = False
        while True:
            line = self._proc.stdout.readline()
            if line == "":
                err = self._proc.stderr.read() if self._proc.stderr else ""
                raise RuntimeError(
                    f"audio server closed stdout mid-caption.\n"
                    f"--- stderr (tail) ---\n{err[-4000:]}"
                )
            stripped = line.strip()
            if stripped == "###CAPTION_START###":
                started = True
                continue
            if stripped == "###CAPTION_END###":
                break
            if started:
                lines.append(line.rstrip("\n"))
        text = "\n".join(lines).strip()
        if text.startswith("__ERROR__:"):
            raise RuntimeError(f"audio server per-item failure: {text}")
        return text

    def close(self) -> None:
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
            self._proc.wait(timeout=30)
        except Exception:
            self._proc.kill()


class AudioVideoSession:
    """Composite session for the audio+video model pairs.

    Holds a video sub-session and an audio sub-session (either in-process or a
    subprocess server), captions each half per segment, and merges them the same
    way ``merge_audio_video_caption`` does downstream — the audio model supplies
    the audio evidence, the video model the visual evidence.
    """

    def __init__(self, video_session, audio_session, source_name: str,
                 audio_model: str, video_model: str) -> None:
        self.video_session = video_session
        self.audio_session = audio_session
        self.source_name = source_name
        self.audio_model = audio_model
        self.video_model = video_model

    def caption(self, segment: Segment, video_path: Optional[str],
                audio_path: Optional[str]) -> CaptionModelOutput:
        video_text = self.video_session.caption_video(video_path or "")
        audio_text = self.audio_session.caption_audio(audio_path or "")
        return CaptionModelOutput(
            modality="audio_video", audio_text=audio_text, video_text=video_text,
            source_caption_model=self.source_name,
            audio_source_model=self.audio_model, video_source_model=self.video_model,
        )

    def close(self) -> None:
        # Close audio first (may be a subprocess we want to tear down cleanly).
        self.audio_session.close()
        self.video_session.close()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_caption_session(caption_model: str, config: RunnerConfig,
                          prompts_dir: Optional[Path] = None):
    """Build the reuse-across-segments session for one caption model.

    Loads the heavy model(s) immediately (so a load failure surfaces up front,
    not mid-run). Caller must ``.close()`` it when done (tears down any
    subprocess audio server).
    """
    spec = cqt.get_model_spec(caption_model)

    if caption_model == "qwen3_omni":
        return Qwen3OmniSession(spec, config, prompts_dir)
    if caption_model == "avocado":
        return Qwen25OmniAVSession(
            spec, config, cqt._AVOCADO_PROMPT,
            system_prompt=cqt._AVOCADO_SYSTEM_PROMPT,
        )
    if caption_model == "timechat":
        return Qwen25OmniAVSession(
            spec, config, cqt._TIMECHAT_PROMPT, fold_timechat=True,
        )
    if caption_model in ("qwen_audio_vl", "af3_vl", "secap_qwen"):
        video_model = config.video_model_path or spec.default_video_model_path
        audio_model = config.audio_model_path or spec.default_audio_model_path
        video_session = Qwen3VLVideoSession(video_model, config)
        if caption_model == "qwen_audio_vl":
            audio_session = QwenOmniCaptionerAudioSession(audio_model)
        elif caption_model == "af3_vl":
            audio_session = SubprocessAudioSession(
                [cqt._AF3_ENV_PYTHON, cqt._AF3_SCRIPT, "--server", audio_model],
                audio_model,
            )
        else:  # secap_qwen
            audio_session = SubprocessAudioSession(
                [cqt._SECAP_ENV_PYTHON, "standalone_inference.py", "--server"],
                audio_model, cwd=str(cqt._SECAP_REPO_DIR / "scripts"),
            )
        return AudioVideoSession(
            video_session, audio_session, source_name=spec.name,
            audio_model=audio_model, video_model=video_model,
        )
    raise ValueError(f"no batch session for caption model {caption_model!r}")
