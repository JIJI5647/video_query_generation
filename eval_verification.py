"""Score verification results against human gold labels, comparing prompts.

Aligns each ``verification_results.jsonl`` with ``gold.jsonl`` by query_id (only
queries that have full human labels are scored), then reports, per results file:

  - Dec.Acc   : 3-class decision accuracy (pass/fail/revise) vs gold decision
  - Accept.F1 : F1 for the "pass" (accept-as-is) class on the final decision
  - FalsePass%: of gold non-pass queries, how many the model passed (safety leak)
  - Rel/Ans/Qual.F1 : per-dimension F1 (positive class = "pass")
  - JSONErr%  : fraction of results that were format/parse failures

Usage:
    python eval_verification.py \
        --gold data/test5_eval/gold.jsonl \
        --results output/verify_*/verification_results.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

_FORMAT_FAIL = "verification output missing or invalid format"


def _read_jsonl(path: Path) -> List[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _prf1(tp: int, fp: int, fn: int):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def _dim_f1(pairs):
    """pairs: list of (gold_bool, pred_bool). F1 with positive class = True (pass)."""
    tp = sum(1 for g, p in pairs if g and p)
    fp = sum(1 for g, p in pairs if (not g) and p)
    fn = sum(1 for g, p in pairs if g and (not p))
    return _prf1(tp, fp, fn)[2]


def _label_from_path(path: Path) -> str:
    # output/verify_p3_fewshot/verification_results.jsonl -> p3_fewshot
    parent = path.parent.name
    return parent[len("verify_"):] if parent.startswith("verify_") else parent


def score_one(gold: Dict[str, dict], results_path: Path) -> dict:
    rows = _read_jsonl(results_path)
    by_id = {r["query_id"]: r for r in rows}
    json_err = sum(1 for r in rows if (r.get("failure_reason") or "") == _FORMAT_FAIL)

    dec_pairs = []          # (gold_decision, pred_decision)
    rel, ans, qual = [], [], []
    matched = 0
    for qid, g in gold.items():
        if g.get("gold_decision") is None:
            continue  # unlabeled
        r = by_id.get(qid)
        if r is None:
            continue
        matched += 1
        dec_pairs.append((g["gold_decision"], r.get("decision")))
        rel.append((bool(g["gold_emotion_relevant"]), bool(r.get("relevance_pass"))))
        ans.append((bool(g["gold_answerability"]), bool(r.get("answerability_pass"))))
        qual.append((bool(g["gold_query_quality"]), bool(r.get("query_quality_pass"))))

    n = len(dec_pairs)
    dec_acc = sum(1 for gd, pd in dec_pairs if gd == pd) / n if n else 0.0
    # Accept ("pass") class F1 on the final decision.
    tp = sum(1 for gd, pd in dec_pairs if gd == "pass" and pd == "pass")
    fp = sum(1 for gd, pd in dec_pairs if gd != "pass" and pd == "pass")
    fn = sum(1 for gd, pd in dec_pairs if gd == "pass" and pd != "pass")
    accept_f1 = _prf1(tp, fp, fn)[2]
    # False pass: of gold non-pass, fraction the model marked pass.
    gold_nonpass = sum(1 for gd, _ in dec_pairs if gd != "pass")
    false_pass = (fp / gold_nonpass) if gold_nonpass else 0.0

    return {
        "label": _label_from_path(results_path),
        "n": n,
        "dec_acc": dec_acc,
        "accept_f1": accept_f1,
        "false_pass": false_pass,
        "rel_f1": _dim_f1(rel),
        "ans_f1": _dim_f1(ans),
        "qual_f1": _dim_f1(qual),
        "json_err": json_err / len(rows) if rows else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Score verification runs vs gold.")
    ap.add_argument("--gold", required=True)
    ap.add_argument("--results", nargs="+", required=True,
                    help="One or more verification_results.jsonl files.")
    ap.add_argument("--csv", default=None, help="Optional path to also write a CSV.")
    args = ap.parse_args()

    gold = {g["query_id"]: g for g in _read_jsonl(Path(args.gold))}
    labeled = sum(1 for g in gold.values() if g.get("gold_decision") is not None)
    print(f"Gold: {len(gold)} queries ({labeled} fully labeled)\n")

    scores = [score_one(gold, Path(p)) for p in args.results]
    scores.sort(key=lambda s: s["label"])

    cols = ["label", "n", "dec_acc", "accept_f1", "false_pass",
            "rel_f1", "ans_f1", "qual_f1", "json_err"]
    head = f"{'prompt':16} {'N':>3} {'Dec.Acc':>8} {'Accept.F1':>9} " \
           f"{'FalsePass':>9} {'Rel.F1':>7} {'Ans.F1':>7} {'Qual.F1':>7} {'JSONErr':>7}"
    print(head)
    print("-" * len(head))
    for s in scores:
        print(f"{s['label']:16} {s['n']:>3} {s['dec_acc']:>8.3f} "
              f"{s['accept_f1']:>9.3f} {s['false_pass']:>9.1%} {s['rel_f1']:>7.3f} "
              f"{s['ans_f1']:>7.3f} {s['qual_f1']:>7.3f} {s['json_err']:>7.1%}")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for s in scores:
                w.writerow({k: s[k] for k in cols})
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
