# Grounding baselines + 313-pool cross-validation (default prompt p0)

Date: 2026-07-15. Zero-shot temporal grounding: give a model the full video +
one emotion query, it outputs a time span; score against a reference.
All numbers use the default prompt p0 (variant sweep was cancelled by user).

## Part 1 — 94 human-gold (真值, models vs human annotation)

| Model | mIoU | R@0.3 | R@0.5 | R@0.7 | parse-fail | audio? |
| --- | --- | --- | --- | --- | --- | --- |
| **Qwen3-VL-30B-A3B** | **0.309** | 48.9% | 28.7% | 16.0% | 0% | no |
| Qwen3-VL-8B | 0.241 | 41.5% | 22.3% | 10.6% | 0% | no |
| Nemotron-3-Nano-Omni | 0.208 | 28.7% | 17.0% | 8.5% | 0% | yes |
| Qwen3-Omni-30B (224p) | 0.109 | 19.1% | 5.3% | 2.1% | 21.3% | yes |
| *prior floor (always front 30%)* | 0.148 | 21.3% | 11.7% | 5.3% | — | — |
| **pipeline proposal (reference)** | **0.903** | 97.9% | 93.6% | 84.0% | 0% | — |

Headline: the best off-the-shelf model (Qwen3-VL-30B, 0.309) barely doubles the
zero-intelligence position prior (0.148) and is 3× below the pipeline's own
fuzzy-time proposals (0.903). No off-the-shelf zero-shot grounder is usable as
a timestamp source on this emotion-query distribution — which is exactly why the
pipeline uses human finalization, not a model, for gold timestamps.

By video length (best-of-3 agreement on the dev split): short ≤60s 0.45,
mid 0.18, long >150s 0.16 — all models collapse on long videos (uniform
frame subsampling loses temporal resolution). 83% of the 313 pool is >150s.

## Part 2 — 313-pool cross-validation (agreement, models vs pipeline proposal)

The 313 nemotron_watch verified-pass queries, split: 100 already human-annotated
(→ 94 gold above), 213 unannotated = the **dev213** cross-check set. Here the
reference is the *pipeline's own proposed time_range* (not human truth), so this
measures agreement, not accuracy.

| Model | mIoU | R@0.5 | parse-fail |
| --- | --- | --- | --- |
| Qwen3-VL-30B | 0.143 | 13.6% | 0.5% |
| Nemotron | 0.078 | 5.2% | 3.3% |
| Qwen3-Omni | 0.009 | 0.5% | 10.8% |

**Critical caveat:** because the baselines themselves are weak (0.2–0.3 vs human
gold), raw "low agreement with the proposal" is dominated by *model* error, not
*proposal* error. 150/213 fall below best-of-3 IoU 0.3, but most of that is the
models being bad, not the proposals being wrong. Raw low-agreement is a noisy
signal and must NOT be read as "150 bad proposals".

### Filtered signal — the review list that actually means something

A proposal is a genuine review candidate only when ≥2 *independent* models agree
with *each other* (IoU ≥ 0.5) yet all disagree with the proposal (best IoU < 0.3).
Two models independently converging on a different location is a real signal;
one weak model scattering is not.

**Result: 14 / 213 queries** (`output/grounding_eval/strong_review_list.jsonl`).

Even these split into two kinds on inspection:
- **Likely genuine proposal error** — e.g. `meld_02_dia224_mm013` "woman appears
  excited" proposal [35–37s] but both models say ~29–37s (wider, earlier onset);
  `emostim_03_MrBeans_mm...` "man in brown suit appears happy" proposal [65–70s]
  but both models firmly at ~24–32s.
- **Multi-moment ambiguity, not error** — e.g. The Shining "man in red shirt
  appears angry while swinging the axe" proposal [105–110s], both models at
  [83–92s]: the film has *repeated* axe-swinging beats, so model and proposal
  likely point at two different valid instances. This is the known
  one-query-many-moments issue, not a wrong timestamp.

## Takeaways

1. Off-the-shelf zero-shot grounding is not viable for emotion-query timestamps
   on this data (best 0.309 mIoU) → validates human finalization in the pipeline.
2. The pipeline's fuzzy proposals (0.903 vs human) are far better than any model,
   confirming the "LLM proposes fuzzy segment-time, human finalizes" design.
3. Cross-validation-by-baseline is a weak QA tool here *because the baselines are
   weak*; only the 14-query mutual-agreement filter is worth human time, and ~half
   of those are multi-moment ambiguity rather than proposal error.

## Part 3 — sliding-window experiment (negative result, 2026-07-15)

Hypothesis: long-video collapse (short 0.45 vs long 0.16) is caused by uniform
frame subsampling destroying temporal resolution → fix by scanning the video in
60s/40s-stride overlapping windows so each is dense, then aggregate.

Result on 41 long-video gold (Qwen3-VL-30B), window-scan vs whole-video:
mIoU 0.207 vs 0.222 (a wash; 13 queries better, 13 worse, 15 same). **Frame
density is NOT the bottleneck.**

Root cause the experiment revealed instead — **query non-uniqueness**:
- 28/41 (68%) queries fire "present" in ≥3 different scattered windows; support
  (windows corroborating one location) is ≤2 for 36/41.
- Monotonic in query specificity: emotion_state fires-in-≥3 at 76% (mIoU 0.169),
  evidence_cue 71% (0.236), explicit_event 50% (0.233). Vaguer query → more
  scattered matches → worse grounding.
- The emotion description ("woman in white appears happy") matches many moments
  because the person is on-screen throughout and the emotion recurs.

**But multi-firing does NOT predict the pipeline's own proposal error**: on the
same 41, pipeline proposals score mIoU 0.95 for high-multi-fire queries and 0.96
for low — essentially identical. The ambiguity is the *baseline model's*
re-discovery problem, not a wrong proposal. This is direct evidence for *why the
caption-first design wins*: the caption stage records which segment produced each
observation, so grounding never has to re-discover "which of the N happy moments"
— the end-to-end models must, and that's where they fail.

Takeaway 4: improving off-the-shelf grounding accuracy on this data has a low
ceiling (window-scan confirms density isn't the lever); the pipeline's advantage
is structural (caption-anchored), not tunable in the grounder.

## Part 4 — does audio help? (2026-07-15)

Split the 94 gold by whether the query TEXT hinges on an audio cue (tone, voice,
says, laugh, cry, gasp, scream, sound, sing, whisper): 7 audio-cue vs 87 not.
Compared the audio model (Nemotron) vs the best vision model (Qwen3-VL-30B):

| subset | Qwen3-VL-30B (vision) | Nemotron (audio) |
| --- | --- | --- |
| audio-cue queries (n=7) | 0.301 | **0.364** |
| no-audio-cue (n=87) | **0.310** | 0.195 |

**Audio helps specifically and only where the query needs it**: on the 7
audio-cue queries Nemotron beats the strongest vision model (4/7 per-query wins:
hearty laugh, cheerful tone ×2, gasp), despite losing by 0.11 overall. On the
other 87 vision-only Qwen3-VL is far ahead. Small n (7%), so directional, but
consistent per-query. Validates the full-modal route as a *targeted* asset for
the audio-dependent slice, not a general win. (Note: 59/94 queries cite audio
evidence in generation metadata, but only 7 genuinely hinge on it in the query —
consistent with the caption-stage over-reading audio tone as visible emotion,
seen in 3 of the 6 false-passes.)

Artifacts: `output/grounding_eval/{gold.jsonl, gold_long.jsonl,
*_p0/predictions.jsonl, qwen3vl30b_window/predictions.jsonl,
strong_review_list.jsonl}`; runners `grounding_baselines/{run_qwenvl.py,
run_nemotron.py, run_window_scan.py, eval_grounding.py}`.
