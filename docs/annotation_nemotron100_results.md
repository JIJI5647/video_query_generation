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

## Queued for event prompt v2.1 (awaiting approval)

1. Salience gate — kill lone-generic-expression events ("expression shifts to smiling"
   → happy; ~16% of events): require ≥2 cue types OR an observable trigger OR marked
   intensity.
2. Relax the ≤4-segment merge cap, or emit chained events for long emotions.
3. explicit_event time_range = minimal action segment(s), not the whole event span.
4. Strip quoted speech from evidence (transcript leak found: "Little pigs, let me
   come in" quoted inside an event's audio evidence).
