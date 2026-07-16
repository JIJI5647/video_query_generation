# Nemotron-watch 100-query human annotation — timestamp agreement results

Date: 2026-07-15 (second, stricter annotation pass; `output/nemotron_watch_100.json`, updated_at 2026-07-15T06:52)

First human-gold measurement of the pipeline's fuzzy segment-level time proposals.
Protocol: human ticks 5s segments per query; agreement = segment-set Jaccard
(equivalent to tIoU on the 5s grid). Queries the human judged un-answerable from the
video are marked `not_groundable` and excluded from IoU stats.

## Headline metrics (94 valid / 6 not_groundable / 100 total)

| Metric | Value | Bar |
| --- | --- | --- |
| mean IoU | **0.899** | ≥ 0.7 usable ✅ (≈ QVHighlights "excellent" 0.9) |
| median IoU | 1.000 | — |
| R@0.5 | **93%** | ≥ 80% usable ✅ |
| R@0.7 | 84% | — |
| exact match (IoU = 1) | 82% | — |
| complete miss (IoU = 0) | 1 | — |
| true one-query-multiple-moments | 2 / 94 | uniqueness defect, rare |

By query type: evidence_cue **0.947** (n=33) > emotion_state 0.898 (n=37) >
explicit_event 0.835 (n=24) — explicit_event is the weak spot (punctual actions
inherit the whole event range).

Delta vs the first annotation pass: mean 0.923 → 0.899, not_groundable 3 → 6,
IoU=0 0 → 1 — the second pass was stricter, conclusions unchanged.

**Verdict: the "LLM proposes query + fuzzy segment-time, human finalizes timestamps"
design is validated** — 82% of proposals matched the human annotation exactly.

## Failure taxonomy

| Pattern | ≈count | Nature / fix |
| --- | --- | --- |
| Long emotion truncated by the ≤4-segment merge cap (e.g. child crying ~50s, model proposes a 20s subset) | ~5 | systematic v2 trade-off → v2.1: relax cap or chained events |
| One-segment boundary jitter (model 1 seg vs human 2) | ~5 | acceptable fuzz; human finalizes anyway |
| Punctual action inherits the whole event range (thumbs-up at s008, range [25,45]) | ~2 | v2.1 one-line rule: explicit_event range = minimal action segments |
| Emotion/cue hallucination that passed the verifier (`not_groundable`: "no sad", "no tears", "no evidence") | **6 (= 6% verifier false-pass)** | recorded; watch |
| Complete miss (model [s015] vs human [s001], "raise his hands to the head") | 1 | isolated |
| True one-query-multiple-moments | 2 | recorded (an earlier "30" count was a mis-read of consecutive 5s ranges = single continuous moments) |

## Root-cause of the 6% verifier false-pass (2026-07-15 investigation)

Traced all 6 not_groundable queries back through query → grounding_evidence →
event → caption. Finding: **the queries are caption-faithful; the hallucination
is at the Qwen3-Omni caption stage.**

- All 6 cite visual evidence that IS present in the caption's `visual_description`
  verbatim (e.g. caption literally says "a single tear visible on his cheek",
  "a look of surprise and concern on his face") — so generation did NOT fabricate;
  it faithfully used the caption.
- The human, watching the actual clip, marks these not-groundable ("No tears",
  "No surprised", "No sad") — i.e. **Qwen3-Omni's visual_description invented
  fine-grained emotion cues that aren't in the video.**
- Two sub-types: (a) fine-grained visual micro-cue fabrication (single tear,
  widened eyes/gasp — the 2 evidence_cue cases); (b) audio tone read as visible
  emotion (caption's only real signal was "voice sounds surprised/strained", but
  the query asks about *appearing* surprised/frustrated — the 3 emotion_state
  cases). One is a pained-expression borderline.
- **Self-reported confidence is useless as a filter**: all 735 captions report
  `confidence=high, evidence_strength=clear` (zero discriminative value).
- The verifier watches the clip but still passes them — it either re-hallucinates
  the same subtle cue or is too lenient on micro-cues.

Implication: the 6% is essentially Qwen3-Omni's caption visual-hallucination rate
on subtle emotion cues, propagated faithfully and invisible to every self-check.
Fix direction (next experiment): an INDEPENDENT cross-model cue check — have a
different, vision-strong model (Qwen3-VL) skeptically verify the specific
fine-grained claim against the clip, since asking the hallucinating model to
self-verify just reproduces the error. NOT filterable on self-confidence.

### Cross-model cue-check experiment (2026-07-15) — refines 6% down to ~3%

Tested an independent skeptical checker (Qwen3-VL-8B, which did NOT write these
captions) on each cited visual cue + its segment clip: "is <cue> ACTUALLY
visible? YES/NO." Test set: the 6 hallucinated + 8 groundable controls (all cite
visual cues, human IoU ≥0.9). `grounding_baselines/cue_check.py`.

Result: **HALLUC caught 3/6, GOOD wrongly flagged 0/8** (zero false positives).
Inspecting the 3 "misses" shows they are not misses but *different error classes*:
- **Clear visual-cue fabrication (3): single tear, pained expression ×2** — the
  checker agrees the cue is absent. Catchable, deployable, 0 collateral damage.
- **Emotion mislabeling (1): MrBean** — the cited cue ("bent over, strained
  expression") IS present (checker confirms); the error is labeling
  exertion-strain as "frustrated". Not a visual hallucination.
- **Subtle-expression model↔human disagreement (2): surprise, widened eyes** —
  BOTH Qwen3-Omni and Qwen3-VL independently read the micro-expression; only the
  human disagrees. Borderline, not a clean fabrication.

Refined conclusion: the true unambiguous caption-visual-hallucination rate is
**~3/94 ≈ 3.2%**, not 6% — the other half is emotion-label interpretation or
borderline micro-expression strictness.

### Scale test KILLS the cross-model gate (2026-07-15) — negative result

Ran the same cue-check over the full pass pool (302 queries) as an automatic
gate. It flagged **175/302 (58%)** — wildly inconsistent with the 6% human
not_groundable rate. Validated against the 95 annotated queries in the pool:
**precision 6.8%, recall 50%** — of 44 flagged, only 3 were truly bad; **41 were
good queries wrongly killed** (46% false-positive rate on good queries).

The earlier "0 false-positives on 8 controls" was small-sample luck (controls
were all from TheShining, a visually clear video). Root cause: a 5-second clip
often doesn't clearly show a fine-grained facial cue (subtle expression, a single
tear), so a strict "only YES if you directly see it" checker defaults to NO on
many real-but-subtle cues. The strictness that catches fabrications destroys
precision. **Do NOT deploy the cross-model cue-check as an auto-gate** — it would
delete ~46% of good queries.

Standing conclusion: caption fine-grained hallucination (~3% clear) is real but
NOT cheaply auto-detectable; the correct catch point is the pipeline's existing
human timestamp-annotation step (which already flags them as not_groundable).
No automatic pre-filter tested beats it.

## Where the pipeline's own 0.903 loss actually is (2026-07-15)

Characterized all 15 low-IoU (<0.7) proposals vs human gold: **13/15 are
"too-narrow" — the proposal is a contiguous SUBSET of a wider human range.** The
pipeline systematically under-emits the range; it almost never over-emits.
Fixing these → overall mIoU 0.903 → **0.979 (+0.076)**. Two sub-patterns:
- **Long-emotion truncation (4, TheChamp)**: prop 15-20s vs human 40-50s — child
  crying ~50s cut by the ≤4-segment (20s) merge cap. Worst cases (IoU 0.30-0.50).
- **One-segment-short (9)**: prop 5s (1 seg) vs human 10s (2 segs), IoU exactly
  0.50 — the emotion spans 2 segments, the pipeline pinned 1.

**Blind post-hoc widening does NOT work**: ±2.5s widening drops mIoU 0.903 → 0.558
(14 better, 79 worse), because 82% of proposals are already exact (IoU=1) and any
blanket transform wrecks them. The fix must use stage-level signal about the true
emotion extent (which captions have), not a geometric post-process. This also
re-confirms proposals are already well-calibrated (narrowing hurt too, earlier).

Concrete recommendation (approval-gated, touches event stage): relax the
≤4-segment merge cap for genuinely long continuous emotions — quantified worth
~+0.03-0.04 mIoU and it fixes the worst cases (IoU 0.30).

### One-segment-short (9 cases) diagnosed = boundary fuzz, NOT a bug (2026-07-15)

Traced all 9: `time_range` correctly matches `segment_ids` (s017→[80,85]) — no
off-by-one code bug. The cause is upstream: the **event stage emits per-segment
events** (meld_02_dia224 has s001/s002/s003 as three separate single-segment
"happy" events) instead of merging adjacent same-emotion segments. Human ticks
the continuous ~10s span (2 segments); the pipeline grounds to the single most
salient 5s segment → IoU exactly 0.50.

This is **defensible boundary calibration, not a defect**: the pipeline's single
segment is the tightest correct localization; the human's 2-segment tick is more
generous, and the human finalizes anyway (the intended division of labor). Blind
widening to "fix" it destroys the 82% already-exact proposals (tested). **No
cheap fix; treat as acceptable fuzz.**

Net: the pipeline's only genuinely-fixable timestamp loss is the 4 TheChamp
cap-truncation cases. Everything else (one-segment fuzz, hallucination,
grounding-model accuracy) is either acceptable, human-caught, or untunable. The
single validated lever remains relaxing the ≤4-segment merge cap for long
emotions.

### Full-pool event-span distribution (2026-07-15)

Across all 153 events in the pool: **111 single-segment (73%)**, 26 two-seg,
6 three-seg, **10 four-seg (6.5%, the cap boundary)**. 30/458 queries (6.6%) are
grounded to cap-boundary events. Confirms at scale: the event stage overwhelmingly
emits tight single-segment events (defensible), and cap-truncation is a small-
prevalence (~6.5%) issue. The cap lives in `mm_event_pilot.py` lines 72/75
(`start + 3`, `lo + 3`) and 103 (`[:4]`) — relaxing requires re-running the
Gemini event stage (approval-gated core-behavior change), so it is proposed, not
applied. Expected payoff ~+0.03-0.04 mIoU, concentrated in the worst (IoU 0.30)
long-emotion cases.

## Queued for event prompt v2.1 (awaiting approval)

1. Salience gate — kill lone-generic-expression events ("expression shifts to smiling"
   → happy; ~16% of events): require ≥2 cue types OR an observable trigger OR marked
   intensity.
2. Relax the ≤4-segment merge cap, or emit chained events for long emotions.
3. explicit_event time_range = minimal action segment(s), not the whole event span.
4. Strip quoted speech from evidence (transcript leak found: "Little pigs, let me
   come in" quoted inside an event's audio evidence).
