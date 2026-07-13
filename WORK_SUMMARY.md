# Work Summary — Research Automation Setup + IoU Eval Design

_Web version (Artifact): https://claude.ai/code/artifact/afe5d1cc-eac1-4be2-b6b6-e250d36c1f2b_
_All paths below are relative to `/work/mzha0323/video_query_generation/` (persistent)._

## Research goal (user)
Improve caption information richness / accuracy → improve downstream **video temporal
grounding**. Caption is the root: wrong caption → everything downstream wrong. First step:
build an **IoU-based grounding evaluation** (pipeline had only measured the verifier layer,
never real grounding IoU).

## What was built this session
1. **Research-automation infra**
   - 5 subagents in `.claude/agents/`: paper-reader, implementer, error-analyst,
     result-analyst, architecture-reviewer.
   - `academic-search` skill in `.claude/skills/` (project-level, persistent).
   - Leader workflow rules appended to `CLAUDE.md`.
   - Notion MCP in `.mcp.json` (project scope; needs `/mcp` OAuth per fresh `$HOME`).
2. **IoU grounding-eval design** → `docs/iou_eval_design.md` (main deliverable, pending approval)
   - Scorer: **training-free Moment-GPT-style proxy** — query-debias (Gemini) + text-embedding
     similarity over our EXISTING segment captions → span. **Minimal/no new GPU.** (Naive
     "ask Qwen for a timestamp" is NOT supported by the literature; UniTime's 74-78% needs LoRA
     training we won't do now.)
   - Metrics: mIoU + R@1,IoU@{0.3,0.5,0.7}.
   - **Mandatory query-agnostic baseline** — 5s-segment gold makes trivial predictions inflate
     IoU; without the baseline the number is meaningless.
   - Label it a **weak proxy grounder** (≈40-60% ceiling), for relative model comparison only.
3. **Eval batch** (`output/eval_pilot_p7_rolecot/`, background): 5 caption models × 19 videos.
   - af3_vl ✅ 13.7% accept · timechat ✅ 20.9% · qwen3_omni running · qwen_audio_vl queued ·
     avocado re-run queued.
   - Key finding: **answerability is the dominant verifier failure (~65%)** — audio-prosody
     emotions aren't visually verifiable in the 5s clip. Verifier-layer, not grounding quality.
   - `emotion_events.py` got a runaway guard (events > 2×captions → resample).

## Where to read what I wrote
| File | What |
|---|---|
| `WORK_SUMMARY.md` | this file |
| `docs/iou_eval_design.md` | **IoU evaluation design** (main deliverable) |
| Artifact (web) | https://claude.ai/code/artifact/afe5d1cc-eac1-4be2-b6b6-e250d36c1f2b |
| `CLAUDE.md` | leader workflow section (end) |
| `.claude/agents/*.md` | 5 subagent definitions |

## Next step (needs approval only if GPU used)
Hand to **implementer**: build `grounding_eval.py` + CLI, IoU smoke on af3_vl + timechat
accepted queries. The embedding-similarity variant is CPU-light (likely no approval needed);
a Qwen sliding-window scorer would be GPU-heavy (needs approval). Confirm and I'll start.
