# IoU-based Temporal Grounding Evaluation — Design

> Status: design complete, **recommendation pending user approval**. Research goal:
> **caption quality → temporal grounding quality**; this doc defines the IoU ruler that makes
> that claim measurable. Synthesized by the leader from 3 paper-reader agents + 1
> result-analyst.

## 1. Why

The pipeline has only ever been measured at the **verifier layer** (acceptance rate,
per-dimension human agreement) — never whether a generated query can actually be **grounded
to the correct time span**. To claim "better captions → better grounding" we need an
**IoU-based grounding metric**. This is the missing ruler.

result-analyst (af3_vl vs timechat) makes the gap concrete: timechat's richer visual captions
give higher acceptance (20.9% vs 13.7%), but that is verifier-layer behavior, **not grounding
quality**.

## 2. Current state

- **No IoU / grounding-eval code exists** (grep: zero hits). Existing eval is verifier-only.
- **gold.jsonl has NO time-span annotation** — only verifier labels + `segment_ids`.

## 3. Data inventory

- Generated queries carry `time_range` **100%** (af3_vl 153/153). Span median 5s (= one 5s
  segment), mean 8.7s, max 55s.
- **⚠️ Self-证 circularity**: that `time_range` is derived from the caption. Usable as a
  *reference* only against an INDEPENDENT scorer's prediction (if caption is wrong, an
  independent model lands elsewhere → low IoU, which is the signal we want).
- gold.jsonl: 55 queries, all have `segment_ids` (mostly 1 segment) → convertible to gold
  span; a small **calibration set**.
- 19 pilot videos, 5s/5s segment grid.

## 4. Scorer feasibility (from paper-reader — the key finding)

**Naive "ask Qwen when does X happen" is NOT supported by the literature.** Two real options:

- **Training-free, Moment-GPT (AAAI'25, arXiv 2501.07972)** — query-debias → per-segment
  caption → **query↔caption embedding similarity + adaptive thresholding** → span. Tops at
  **R1@0.5 ≈ 38-58%** on event-centric benchmarks. We can replicate it with components we
  already run: **reuse our existing OmniCaption segment captions + Gemini query-debias + a
  text-embedding similarity**, i.e. **almost no new GPU inference**. Cheapest, most defensible
  first cut.
- **Fine-tuned, UniTime (arXiv 2506.18883)** — R1@0.5 up to 74-78%, but needs LoRA on ~208K
  grounding queries with a timestamp-token-interleaving prompt. Training infra we don't have;
  not for now.
- **Reliability ceiling**: even best zero-shot ≈ 40-60% IoU@0.5 — comparable to the error we're
  measuring. So label the scorer a **"weak proxy grounder", NOT "pseudo-gold"**; use for
  **relative / coarse** caption-model ranking, never as truth. TimeMarker (released 8B weights)
  is a heavier fallback.

## 5. Metric definitions (UniVTG + Debiased-TSG)

- **IoU** of predicted span P and reference G = `|P∩G| / |P∪G|` (seconds).
- **R@n,IoU@m**: top-n spans; "hit" if ANY has IoU ≥ m. n=1 → best span's IoU ≥ m.
- **Thresholds**: m ∈ {0.3, 0.5, 0.7}, n ∈ {1, 5}.
- **mIoU** = mean top-1 IoU over queries — the single most informative summary.
- **Headline**: mIoU (primary) + R@1,IoU@{0.3,0.5,0.7}.

## 6. Pitfalls to guard against (CRITICAL — Debiased-TSG)

- **Moment-location / length bias**: models score high by exploiting that gold moments cluster
  at certain locations/lengths, without understanding the query.
- **Our specific risk**: 5s-segment grid → many reference spans = one 5s chunk → a trivial
  "predict a 5s chunk" hits by luck and inflates IoU.
- **MANDATORY control**: report a **query-agnostic baseline** (whole-video / random 5s span).
  If our scorer's IoU isn't clearly above it, the metric measures the prior, not grounding.
  **Without this baseline the IoU number is meaningless.**
- Single-moment gold assumed; multi-segment emotion events need mIoU-over-moments — flag, later.

## 7. Recommendation (leader — pending user approval)

**Primary = training-free Moment-GPT-style proxy grounder reusing our existing captions, run
for RELATIVE caption-model comparison, WITH the mandatory query-agnostic baseline.**

1. **Scorer**: query-debias (Gemini) → text-embedding similarity between the query and each
   segment's OmniCaption fields → adaptive threshold/merge → predicted span. **No new GPU
   inference** (reuses captions we already have) → likely does NOT need GPU approval.
2. **Reference span**: the query's own `time_range`; cross-check on the 55-query calibration
   set (`segment_ids` gold) to confirm the scorer isn't garbage.
3. **Baseline**: whole-video + random 5s span (query-agnostic). REQUIRED.
4. **Metrics**: mIoU + R@1,IoU@{0.3,0.5,0.7}, per caption model.
5. **First run (smoke)**: af3_vl + timechat accepted queries (21 + 39) — test whether
   timechat's acceptance edge translates to a grounding-IoU edge.
6. **Then**: extend to qwen3_omni / qwen_audio_vl / avocado as their eval exports.

**Why this over UniTime/naive-Qwen**: no training, no new model, minimal/no GPU, runs on the
current 5 models, directly answers caption→grounding. Its noise is acceptable because we use it
for *relative* comparison on a shared scorer — reported as a weak proxy, not truth.

## 8. Implementation plan (needs approval only if GPU is used)

Hand to **implementer**:
1. `emotion_query_pipeline/grounding_eval.py`: query-debias + embedding-similarity span
   predictor over existing segment captions; IoU / R@n,IoU@m / mIoU + query-agnostic baseline.
2. CLI `run_grounding_eval.py --queries-dir output/eval_pilot_p7_rolecot/<model> ...`.
3. Smoke on af3_vl + timechat accepted queries; sanity-check on the 55-query calibration set.
4. Report per-model mIoU + R@1 table with baseline row.

The embedding-similarity variant is CPU/light; if we instead try a Qwen sliding-window scorer
(heavier), that GPU step needs approval.
