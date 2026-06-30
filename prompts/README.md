# Prompts index

All prompt files live **flat** in this folder. Do NOT move them into subfolders:
code loads them by bare filename, and `{{include: ...}}` directives resolve
against this directory (moving the included `rule_*.txt` / `verification_rules.txt`
would break `--using-prompt` staging in `run_verification.py`).

`{{include: file.txt}}` inlines another file at load time (see
`io_utils.load_prompt_template`), so shared rules live in one place.

---

## Verification ablation experiment

Driven by `run_verification.py` / `run_verification_sweep.sh`, scored by
`eval_verification.py`. Every variant emits the **same JSON schema**
(`relevance_pass` / `answerability_pass` / `query_quality_pass` + `failure_reason`);
`decision` is derived in code, never output by the model.

### Per-dimension architecture (default, `MODE=perdim`)
Each strategy variant (p0..p8) is judged as **3 separate inferences** — one per
dimension. **relevance & query_quality are judged from the query TEXT ONLY**
(no video); **answerability watches the clip**. Run via
`--per-dimension --variant <name>`.

The full prompt for every (variant x dimension) is a **file** under `perdim/`:
`perdim/vdim_<variant>_<slug>.txt` (slug = relevance | answerability |
query_quality). Each file composes itself via `{{include}}` of the shared
fragments below — the experiment design lives in these files, **not** in Python.
The code only loads the named file and fills `{video_id}` / `{round_index}` /
`{queries_json}`. (The one thing in code, `_DIM_NEEDS_VIDEO`, just decides whether
to attach the clip — relevance/quality: no, answerability: yes.)

| Variant | rule | role | few-shot | CoT |
|---------|:----:|:----:|:--------:|:---:|
| p0_norule | – | – | – | – |
| p1_rule | ✓ | – | – | – |
| p2_role | ✓ | ✓ | – | – |
| p3_fewshot | ✓ | – | ✓ | – |
| p4_zscot | ✓ | – | – | ✓ (steps) |
| p5_fewshotcot | ✓ | – | ✓ | ✓ |
| p6_rolefewshot | ✓ | ✓ | ✓ | – |
| p7_rolecot | ✓ | ✓ | – | ✓ |
| p8_rawcot | ✓ | – | – | ✓ (bare) |

Shared fragments included by the `perdim/` files:
- `strat_role.txt`, `strat_rawcot.txt` — generic role / bare "think step by step".
- `cot_{relevance,answerability,query_quality}.txt` — per-dimension CoT step.
- `fewshot_{relevance,answerability,query_quality}.txt` — per-dimension examples.
- `rule_{relevance,answerability,query_quality}.txt` — per-dimension rule.
- `rule_suggested_revision.txt` — suggested_revision policy (combined prompts only).

To add/adjust a variant: edit (or copy) the three `perdim/vdim_<variant>_*.txt`
files. To change a rule everywhere: edit the relevant `rule_*.txt` once.

### Combined prompts (legacy, `MODE=combined`)
One inference judges all 3 dimensions (watching the clip). Run via
`--using-prompt prompts/<file>`.

| File | Variant |
|------|---------|
| `verification_prompt.txt` | p1_rule (default) |
| `verification_prompt_p0_norule.txt` … `_p8_rawcot.txt` | p0, p2–p8 |
| `verification_rules.txt` | composes the 4 `rule_*.txt` (included by the above) |

**To change a rule: edit the relevant `rule_*.txt` once** — it propagates to both
the per-dimension composer and the combined variants (via `verification_rules.txt`).

---

## Main pipeline prompts (run_pipeline.py — not the verification experiment)

| File | Stage |
|------|-------|
| `caption_prompt.txt` | Gemini caption backend |
| `omni_caption_prompt.txt` | Qwen3-Omni caption backend (observation-only) |
| `emotion_event_prompt.txt` | Gemini emotion-event stage (8 labels) |
| `generation_prompt.txt` | query generation from events + captions |
| `rewrite_prompt.txt` | query rewrite (used by `rewriting.py`) |
