# Progress log

Live status of in-progress work. Update as tasks complete; this is a status log,
not a reference doc (see `docs/caption_query_test.md` / `README.md` for those).

## This machine: `$HOME` is wiped on restart, only `/work/mzha0323/...` persists

Machine-specific, not general repo guidance (other environments running this repo may
differ) — noting it here rather than in `CLAUDE.md`. `df -h` on this server shows `$HOME`
(`/home/dgxuser`) on the container's `overlay` root fs and `/work` on a separate persistent
Lustre mount. `env.sh` (API keys), `conda_envs/`, `hf_cache/` (`HF_HOME`), `pip_cache/` are
already kept under the project dir instead of their usual home-directory defaults for this
reason. Also affects `~/.gitconfig`, `~/.ssh/`, `~/.netrc`, `~/.git-credentials` — git
identity on this machine is set via `git config --local` (this repo's own `.git/config`,
under `/work`) instead of `--global`, and the push SSH key lives at
`/work/mzha0323/.ssh_persist/id_ed25519` (outside the git working tree) with
`core.sshCommand` pointed at it.

Same problem hit **Claude Code itself**: the native install
(`~/.local/share/claude/versions/...`), login credentials (`~/.claude/.credentials.json`),
and config (`~/.claude.json`, `~/.claude/settings.json`, `~/.claude/plugins/`) all live
under `$HOME` too, so a restart used to mean reinstalling Claude Code from scratch AND
re-running the interactive login every time. Fixed the same way — snapshotted to
`/work/mzha0323/.claude_persist/` (outside any git repo, since `.credentials.json` is a
live auth token and must never be committed):

- `/work/mzha0323/.claude_persist/backup.sh` — re-run any time (safe while Claude Code is
  running; only copies, never touches the live `$HOME` files) to refresh the snapshot,
  e.g. after `claude update` or re-logging in.
- `/work/mzha0323/.claude_persist/restore.sh` — run ONCE after a container restart,
  **before** starting `claude` for the first time in the fresh `$HOME`, to restore the
  install + login from the snapshot (no reinstall, no re-login).

`env.sh` runs this automatically so `source env.sh` alone is enough: restores from the
snapshot if one exists and `$HOME` doesn't have credentials yet, or just refreshes the
snapshot if already logged in. If neither a snapshot nor a local `claude` binary exists at
all (truly first-ever setup, no prior backup ever made), it falls back to running the
official installer (`curl -fsSL https://claude.ai/install.sh | bash`) non-interactively —
login is still a one-time manual step in that case (no OAuth token exists anywhere to
restore), after which run `backup.sh` once to snapshot it for future restarts.

Deliberately does NOT snapshot conversation history / live session state
(`~/.claude/projects/`, `sessions/`, `session-env/`, `shell-snapshots/`, `tasks/`,
`file-history/`, `plans/`, `downloads/`, `cache/`, `backups/`) — those are per-session
working state, not the install/login this exists to protect, and copying the *currently
open* session's transcript mid-write would risk losing its tail end. Losing old
conversation history/resumability on restart is a known, accepted gap here — not what
this solves.

**Exception: the agent's persistent memory** (`~/.claude/projects/<slug>/memory/`) is
NOT session transcript despite living under the excluded `~/.claude/projects/` path — it's
durable cross-conversation knowledge the agent writes to directly, so it needed real
persistence rather than being caught by the exclusion above. Fixed by moving it out from
under `$HOME` entirely: it now lives at
`/work/mzha0323/.claude_persist/agent_memory/<slug>/` and
`~/.claude/projects/<slug>/memory` is a symlink to that (done once directly; `restore.sh`
recreates the symlink for every project dir found under `agent_memory/` after a restart).
Being a symlink (not a copy), writes land on `/work` immediately — `backup.sh` doesn't need
to do anything for it.

## Verification: Qwen3-Omni-Thinking sweep

Ablation of the 9 per-dimension verification-prompt variants (p0-p8), run with the
`Qwen/Qwen3-Omni-30B-A3B-Thinking` reasoning checkpoint as the verify backend
(`--qwen-model-path`, `--qwen-max-tokens 8192`). Scored against `data/test5_eval/gold.jsonl`
(55 gold-labeled queries) into `output/verify_metrics_thinking.csv`.

| Variant | Status | n | dec_acc | accept_f1 | false_pass | wall time |
|---|---|---|---|---|---|---|
| p0_norule | done | 55 | 0.618 | 0.615 | 47.2% | 2.63h |
| p1_rule | done | 55 | 0.709 | 0.706 | 38.9% | 3.28h |
| p2_role | done | 55 | 0.673 | 0.638 | 36.1% | 4.62h |
| p3_fewshot | done | 55 | 0.618 | 0.630 | 50.0% | 5.40h |
| p4_zscot | done | 55 | 0.673 | 0.667 | 41.7% | 5.88h |
| p5_fewshotcot | done | 55 | 0.673 | 0.667 | 41.7% | 4.56h |
| p6_rolefewshot | done | 55 | 0.655 | 0.654 | 44.4% | 3.75h |
| p7_rolecot | done | 55 | 0.600 | 0.553 | 41.7% | 5.48h |
| p8_rawcot | **running** (started 2026-07-07 01:29:52, last variant of `run_p4_p8_thinking_sweep.sh`, pid file `p4_p8_sweep.pid`) | - | - | - | - | in progress (5h11m so far) |

Wall time = per-variant sweep duration (all 5 videos, 3 dimensions each, `--parallel 1`), read off
`logs/p{0_p8,1_p3,4_p8}_thinking_sweep.log` timestamps — total compute so far for p0-p7:
**35.6h**. No correlation between strategy complexity and runtime (p4_zscot, the longest at
5.88h, scores mid-pack; p1_rule, 2nd-shortest, scores best) — runtime is driven by
per-video/per-query variance in the Thinking model's CoT length, not variant design.

p0-p7 scored into `output/verify_metrics_thinking_p0_p7.csv` (2026-07-07). Best so far: **p1_rule**
(dec_acc 0.709, accept_f1 0.706, lowest false_pass 38.9%) — none of p4-p7 beat it. Worst: p7_rolecot
(dec_acc 0.600). Full 9-variant table + final `verify_metrics_thinking.csv` once p8 finishes.

p4-p8 launched as one background sweep (`nohup bash run_p4_p8_thinking_sweep.sh > logs/p4_p8_thinking_sweep.log 2>&1 &`),
same settings as p0-p3 (`--qwen-model-path Qwen/Qwen3-Omni-30B-A3B-Thinking --qwen-max-tokens 8192 --parallel 1`).
Per-variant progress in `logs/verify_sweep/p{4,5,6,7,8}_*.log`, overall sequencing in
`logs/p4_p8_thinking_sweep.log`. Based on p0-p3 timing (2.6h-5.4h each, increasing), expect
this to take roughly 20-25h wall time total. Stale `output/verify_p4_zscot` /
`verify_p5_fewshotcot` from an earlier **Instruct**-model (not Thinking) run were archived to
`*_instruct_bak` before this sweep started — do not confuse the two.

First launch attempt (05:47) used the wrong Python (`/opt/conda/bin/python`, base env without
`transformers` installed) and silently produced all-`fail` garbage results in ~3s for all 5
variants — caught by checking log content instead of just exit code, cleaned up, and relaunched
via the project's `conda_envs/video_env/bin/python` (a full conda env directory, no
`bin/activate`, so the script sets `PYTHON="$(pwd)/conda_envs/video_env/bin/python"` and calls
it directly rather than sourcing an activate script).

Once all 9 variants are done, score with:
```
python eval_verification.py --gold data/test5_eval/gold.jsonl \
  --results output/verify_p{0,1,2,3,4,5,6,7,8}_*/verification_results.jsonl \
  --csv output/verify_metrics_thinking.csv
```

**Fixed along the way:** `verify_queries_per_dimension` (`emotion_query_pipeline/verification.py`)
used to let one query's exhausted-retries failure abort an ENTIRE video's results (all its
other queries lost too) — this is why `p1_rule`'s first pass only had 40/78 results (the
`moviechat_05_30` video, 38 queries, kept failing on one bad query and losing all of them,
twice). Added per-chunk try/except so a failed query is marked `fail` with reason
`"<dim>: call failed"` instead of crashing the whole video (see `tests/test_verification_perdim.py::test_one_query_failure_does_not_lose_the_rest`).
The missing 38 `moviechat_05_30` queries for `p1_rule` were then re-run in isolation
(`output/verify_p1_rule_missing/`) and merged into `output/verify_p1_rule/verification_results.jsonl`,
giving the complete n=55 numbers above.

**Next:** run p4_zscot through p8_rawcot with the same Thinking checkpoint so all 9
variants are comparable (currently only 4/9 done).

## Caption model integration tests (`run_caption_query_test.py`)

Testing that each of the 6 supported `--caption-model` backends can actually run
end-to-end (caption → Gemini emotion-event → Gemini query-generation), not just that the
runner code is wired (see `docs/caption_query_test.md` for the model matrix). Test clip:
`data/pilot_study/emostim_07_TheSilenceOfTheLambs_a_clip_2.mp4` (12.7s, shortest available),
`--max-segments 1`. Run ONE AT A TIME (not concurrently) to stay inside the 8 CPU / 32GB
RAM allocation — see the resource-limits note in `CLAUDE.md`.

| Model | Status | Result |
|---|---|---|
| `qwen3_omni` | not run via this script (but exercised extensively via the main pipeline) | - |
| `avocado` | **done** | 1 caption, 1 emotion event, 3 queries generated. `output/caption_query_tests/avocado/` |
| `timechat` | **done — found a real bug, see below** | 1 caption (`caption_status: salvaged`), 0 emotion events, 0 queries. `output/caption_query_tests/timechat/` |
| `qwen_audio_vl` | **done** | 1 caption, 1 emotion event, 3 queries generated. `output/caption_query_tests/qwen_audio_vl/` (423.5s caption call — largest so far: downloads + loads both Qwen3-Omni-Captioner audio and Qwen3-VL-8B video) |
| `af3_vl` | **done** | 1 caption, 1 emotion event, 2 queries generated. `output/caption_query_tests/af3_vl/` (caption call itself 231.7s, but ~10h wall time — first-time download of `nvidia/audio-flamingo-3-hf` weights, ~16GB added to `hf_cache`; two repo names cached (`audio-flamingo-3` and `audio-flamingo-3-hf`), possibly a redundant pull) |
| `secap_qwen` | **cancelled** | run was cut off mid-inference (loading Qwen3-VL, about to sample video frames) when the session was reclaimed ~2026-07-06 03:01; not resumed, dropped from this test pass. `logs/caption_query_tests/secap_qwen.log` has the partial log. |

### Bug found: TimeChat's JSON-array output isn't normalized correctly

TimeChat-Captioner-GRPO-7B returns a **JSON array of per-timestamp scene dicts**
(`[{"timestamp": "00:00-00:04", "segment_detail_caption": "...", "storyline": "...",
"acoustics_content": "...", ...}, {"timestamp": "00:05-00:09", ...}, ...]`) — the model's
whole selling point is multi-scene, timestamped captions for clips up to ~1 min. But
`normalize_to_omni_caption` (`emotion_query_pipeline/caption_query_test.py`) only special-cases
a JSON **object** (`_try_parse_json_object` looks for the first `{`); for a JSON array it
still finds a `{` (the first array element) and `json.raw_decode` happily parses JUST that
first dict, discarding everything after it. That dict's keys (`timestamp`,
`segment_detail_caption`, `storyline`, `acoustics_content`, ...) don't match any recognized
`OmniCaption` field, so the "no usable content" branch fires and the caption is marked
`caption_status: salvaged` with `temporal_description` set to a **500-char truncated preview
of the raw array text** — which for this test clip cuts off after the FIRST timestamp chunk
(a establishing shot, no fear content) and never reaches the 2nd/3rd chunks that describe the
woman's fear and drawing a gun. That's why the emotion-event stage found nothing: the actual
emotional content never made it past normalization, not because TimeChat failed to caption it
(its raw output clearly captured the fear/gun content — see
`output/caption_query_tests/timechat/raw_caption_output.json`).

**Fix needed** (not yet done): give `normalize_to_omni_caption` (or a TimeChat-specific
pre-step in `_run_timechat`) array-aware handling — parse a JSON-array raw output, and fold
each element's `segment_detail_caption` (+ `storyline`/`acoustics_content`) into
`temporal_description` with its timestamp prefixed, rather than truncating the raw string.
Ask before implementing — needs a decision on exactly which per-timestamp fields to keep and
how to format them into one `temporal_description` string.

Each run downloads its model weights on first use (into `HF_HOME=/work/mzha0323/hf_cache`,
NOT the default `~/.cache`) — first-run wall time is dominated by the download, not
inference (`avocado`'s actual caption call was 126.8s out of ~1h23m total). GPU memory and
cgroup RSS both return to baseline after each run exits (verified via `nvidia-smi` /
`/sys/fs/cgroup/memory.current` between runs).

## Caption-query-test rework: 3 independent stages + real 5s multi-segment sweep

`run_caption_query_test.py` was cancelled and replaced by 3 independent, cacheable
CLI stages that each read the prior stage's output dir (see `docs/caption_query_test.md`):
`run_caption_generation_test.py` → `run_query_generation_test.py` → `run_evaluation_test.py`.

Two real wiring bugs were found and fixed before the 5s sweep could mean anything:

1. **Multi-segment mode cut real clips but never fed them to the model.** `--max-segments
   > 1` already called `extract_segment_clips` (real ffmpeg cuts, `segment.clip_path` set),
   but the main loop still passed `video_path=args.video` (the whole original file) to
   every segment's caption call — so every prior "multi-segment" test silently captioned
   the same whole video 3x instead of 3 distinct 5s clips. Fixed: use `seg.clip_path`.
2. **`audio_video`-kind models (`qwen_audio_vl`/`af3_vl`/`secap_qwen`) were hard-blocked
   from `--max-segments > 1`.** Fixed by extracting a per-segment audio `.wav` directly
   from each cut video clip (`clip_extractor.extract_audio_track`) — no separate
   pre-split whole-audio input needed.

Also fixed the TimeChat JSON-array bug documented above: `_run_timechat` now folds
per-timestamp scenes (`segment_detail_caption` + `storyline` + `acoustics_content`) into
one plain-text `temporal_description` before normalization, instead of the generic
normalizer truncating to the first scene.

**5s sweep** (5 of 6 models — `secap_qwen` excluded, still cancelled from the earlier
session-reclaim interruption), same test clip, `--max-segments 3` (3 real segments:
5s/5s/2.7s), full 3-stage run each, one model at a time:

| Model | Segments | Emotion events | Queries generated | Verify: accepted/discarded | Output |
|---|---|---|---|---|---|
| `qwen3_omni` | 3 (real, distinct captions) | 1 | 3 | 2 / 1 (66.7%/33.3%) | `output/caption_query_tests_5s/qwen3_omni/` |
| `avocado` | 3 | 2 | 4 | 2 / 2 (50%/50%) | `output/caption_query_tests_5s/avocado/` |
| `timechat` | 3 (`caption_status: normalized`, was `salvaged`) | 2 | 4 | 1 / 3 (25%/75%) | `output/caption_query_tests_5s/timechat/` |
| `qwen_audio_vl` | 3 (+ 3 distinct extracted audio slices) | 3 | 4 | 1 / 3 (25%/75%) | `output/caption_query_tests_5s/qwen_audio_vl/` |
| `af3_vl` | 3 (+ 3 distinct extracted audio slices) | 1 | 2 | 0 / 2 (0%/100%) | `output/caption_query_tests_5s/af3_vl/` |

All 5 models ran the full 3-stage pipeline over real 5s segments without crashing, and
TimeChat's caption content is no longer being thrown away. Verify accept-rates are not a
quality signal here (5 gold-free segments, no comparable baseline) — this is still a
plumbing/integration check, not a benchmark.

## Verification: Nemotron-3-Nano-Omni + TensorRT-LLM (attempted — blocked by GPU driver)

Goal: run the same p0-p8 per-dimension verifier ablation on NVIDIA's
`Nemotron-3-Nano-Omni-30B-A3B-Reasoning` (a Mamba2-hybrid-MoE omni model with
C-RADIOv4-H vision + Parakeet audio encoders, ~3B active of 31B; a *reasoning*
checkpoint, so a natural counterpart to the Qwen3-Omni-**Thinking** sweep) served on
**TensorRT-LLM**, the efficient NVIDIA inference framework.

**Outcome: cannot run TensorRT-LLM (nor the newest vLLM) for this model on this machine —
hard blocker is the host GPU driver, which cannot be upgraded from inside the container.**

- The model's serving support is brand new and lands only in **CUDA-13-era** stacks:
  - TensorRT-LLM: the omni arch + `nano-v3` reasoning parser need **1.3.0rc13+**, whose
    wheels hard-depend on `cuda-python>=13`, `nvidia-nccl-cu13`, `torch>=2.10` (CUDA 13).
  - vLLM: Nemotron-3-Nano-Omni support landed in **vLLM 0.20.0**, which pins
    `torch==2.11.0` (also CUDA 13). Our `conda_envs/vllm_env` has vLLM 0.14.0, whose model
    registry has NO `NemotronH_Nano_Omni_Reasoning_V3` entry.
- This host: **driver 550.163.01** (CUDA 12.6, Hopper H200, cc 9.0), **no docker**, no root
  to touch the kernel driver. Per NVIDIA's forward-compatibility matrix, **CUDA 13.x
  forward-compat needs a base driver >= R580** (R570 is the floor for `cuda-compat-13`);
  R550 does not qualify. So no CUDA-13 runtime — trtllm 1.3 / vLLM 0.20 / torch 2.10-2.11 —
  can initialize a CUDA context here. The docker image path (`nvcr.io/.../tensorrt-llm`) is
  also out (no docker).

**What was still done (the genuine attempt + a drop-in integration ready for an R580+ box):**

1. **Downloaded** `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8` (33 GB, 4 safetensors
   + remote code) into `HF_HOME=/work/mzha0323/hf_cache`. FP8 (modelopt) is the right
   precision for Hopper (NVFP4's FP4 tensor cores are Blackwell-only; BF16 would be 62 GB).
2. **Installed TensorRT-LLM 1.3.0rc15** into `/work/mzha0323/trtllm_venv` (pip, from
   `pypi.nvidia.com`) to attempt it for real — see `logs/trtllm_install.log`. [RUNTIME
   RESULT — trtllm-serve / import outcome: TO BE FILLED once the install finishes.]
3. **Engine-agnostic integration** (works unchanged for `trtllm-serve` OR `vllm serve`,
   since both expose the identical OpenAI `/v1/chat/completions` API — switching engines is
   a one-line change to the server launch command only):
   - `emotion_query_pipeline/nemotron_client.py` — `NemotronOpenAIClient`, a duck-typed
     `BaseLLMClient` that POSTs to the OpenAI-compatible endpoint. Takes LOCAL clip path(s)
     as `video_uri` (like `QwenOmniLLMClient`, no upload), converts to `file://` URIs,
     fires a chunk of requests concurrently (HTTP analog of the Qwen batched forward), and
     robustly extracts JSON (strips `<think>` blocks / fences, `raw_decode` from first `{`).
     A query that exhausts retries returns `{}` (read as a fail-safe "invalid format" by the
     per-dimension verifier) so one bad query never loses the rest of its chunk — same
     blast-radius guarantee as the Qwen path.
   - `run_verification.py` — new `--verify-rewrite-backend nemotron` with
     `--nemotron-{base-url,model,max-tokens,no-thinking}`; `use_local_clips` now covers both
     the Qwen engine and the Nemotron server.
   - `run_trtllm_serve.sh` — launches `trtllm-serve` (nano-v3 reasoning parser, FP8).
   - `run_nemotron_sweep.sh` — the p0-p8 sweep against the server (mirrors
     `run_verification_sweep.sh`); score with `eval_verification.py` into
     `output/nemotron_sweep/verify_metrics_nemotron.csv`, comparable to the Qwen Thinking
     table above. Client logic unit-checked offline (no server needed).

**To actually run it:** on a host with driver >= R580 (or R570 + cuda-compat-13), start
`bash run_trtllm_serve.sh`, wait for the server, then `bash run_nemotron_sweep.sh`. No code
changes needed. On this R550 box the sweep cannot run against TensorRT-LLM.
