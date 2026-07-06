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

Deliberately does NOT snapshot conversation history / live session state
(`~/.claude/projects/`, `sessions/`, `session-env/`, `shell-snapshots/`, `tasks/`,
`file-history/`, `plans/`, `downloads/`, `cache/`, `backups/`) — those are per-session
working state, not the install/login this exists to protect, and copying the *currently
open* session's transcript mid-write would risk losing its tail end. Losing old
conversation history/resumability on restart is a known, accepted gap here — not what
this solves.

## Verification: Qwen3-Omni-Thinking sweep

Ablation of the 9 per-dimension verification-prompt variants (p0-p8), run with the
`Qwen/Qwen3-Omni-30B-A3B-Thinking` reasoning checkpoint as the verify backend
(`--qwen-model-path`, `--qwen-max-tokens 8192`). Scored against `data/test5_eval/gold.jsonl`
(55 gold-labeled queries) into `output/verify_metrics_thinking.csv`.

| Variant | Status | n | dec_acc | accept_f1 | false_pass |
|---|---|---|---|---|---|
| p0_norule | done | 55 | 0.618 | 0.615 | 47.2% |
| p1_rule | done | 55 | 0.709 | 0.706 | 38.9% |
| p2_role | done | 55 | 0.673 | 0.638 | 36.1% |
| p3_fewshot | done | 55 | 0.618 | 0.630 | 50.0% |
| p4_zscot ... p8_rawcot | **running** (started 2026-07-06 05:49, `run_p4_p8_thinking_sweep.sh`, pid file `p4_p8_sweep.pid`) | - | - | - | - |

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
