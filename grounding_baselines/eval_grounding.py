"""Score grounding predictions against the human gold set.

Protocol (verified against TimeChat/VTimeLLM/QVHighlights official eval code):
  - top-1 single prediction per query; parse failure counts as IoU=0, stays in
    the denominator (parse-failure rate reported separately)
  - multi-span gold: IoU = max over gold spans (QVHighlights convention)
  - primary: continuous-seconds R@1@{0.3,0.5,0.7} + mIoU
  - secondary: 5s-grid segment Jaccard (DiDeMo-style; prediction snapped to
    the 5s grid) — directly comparable to the pipeline's own annotation metric
  - 95% percentile bootstrap CI (1000 resamples) on mIoU and R@1 metrics

Predictions file: jsonl rows {query_id, pred: [start, end] | null}.
  python grounding_baselines/eval_grounding.py --gold output/grounding_eval/gold.jsonl \
      --pred output/grounding_eval/qwen25vl/predictions.jsonl [--label qwen25vl]
Multiple --pred allowed; prints one table row per pred file.
"""
import argparse, json, random
from pathlib import Path

SEG = 5.0
THRESHOLDS = (0.3, 0.5, 0.7)


def iou_1d(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def max_iou(pred, gold_ranges):
    return max(iou_1d(pred, g) for g in gold_ranges)


def to_grid(rng, duration):
    """Continuous [s,e] -> set of 5s segment indices (snap: any overlap >=40%
    of the segment or >=half the range counts; simple round to grid edges)."""
    s = max(0.0, min(rng[0], duration))
    e = max(0.0, min(rng[1], duration))
    if e <= s:
        return set()
    first = int(s // SEG)
    last = int((e - 1e-9) // SEG)
    segs = set()
    for i in range(first, last + 1):
        seg_s, seg_e = i * SEG, (i + 1) * SEG
        overlap = min(e, seg_e) - max(s, seg_s)
        if overlap >= 0.5 * min(SEG, e - s):
            segs.add(i)
    return segs or {first}


def grid_jaccard(pred, gold_ranges, duration):
    p = to_grid(pred, duration)
    best = 0.0
    for g in gold_ranges:
        gg = to_grid(g, duration)
        if p or gg:
            best = max(best, len(p & gg) / len(p | gg))
    return best


def score(rows):
    ious = [r["iou"] for r in rows]
    n = len(ious)
    res = {"n": n, "mIoU": sum(ious) / n,
           "parse_fail": sum(1 for r in rows if r["pred"] is None) / n,
           "grid_jaccard": sum(r["grid_j"] for r in rows) / n}
    for t in THRESHOLDS:
        res[f"R@{t}"] = sum(1 for x in ious if x >= t) / n
    return res


def bootstrap_ci(rows, metric_fn, n_boot=1000, seed=0):
    rng = random.Random(seed)
    n = len(rows)
    vals = sorted(metric_fn([rows[rng.randrange(n)] for _ in range(n)])
                  for _ in range(n_boot))
    return vals[int(0.025 * n_boot)], vals[int(0.975 * n_boot)]


def evaluate(gold, preds_by_id, label):
    rows = []
    for g in gold:
        p = preds_by_id.get(g["query_id"])
        pred = p.get("pred") if p else None
        if pred and pred[1] > pred[0]:
            iou = max_iou(pred, g["gold_ranges"])
            gj = grid_jaccard(pred, g["gold_ranges"], g["duration"])
        else:
            pred, iou, gj = None, 0.0, 0.0
        rows.append({**g, "pred": pred, "iou": iou, "grid_j": gj})
    res = score(rows)
    lo, hi = bootstrap_ci(rows, lambda rs: sum(r["iou"] for r in rs) / len(rs))
    res["mIoU_ci"] = (lo, hi)
    lo, hi = bootstrap_ci(rows, lambda rs: sum(1 for r in rs if r["iou"] >= 0.5) / len(rs))
    res["R@0.5_ci"] = (lo, hi)
    # per-type breakdown
    by_type = {}
    for t in sorted(set(r["query_type"] for r in rows)):
        sub = [r for r in rows if r["query_type"] == t]
        by_type[t] = {"n": len(sub), "mIoU": sum(r["iou"] for r in sub) / len(sub)}
    return res, by_type, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="output/grounding_eval/gold.jsonl")
    ap.add_argument("--pred", nargs="+", required=True)
    ap.add_argument("--dump-rows", action="store_true",
                    help="write per-query scored rows next to each pred file")
    args = ap.parse_args()

    gold = [json.loads(l) for l in open(args.gold) if l.strip()]
    hdr = f"{'model':<22}{'n':>4}{'mIoU':>7}{'R@0.3':>7}{'R@0.5':>7}{'R@0.7':>7}{'gridJ':>7}{'fail':>6}  mIoU 95%CI      R@0.5 95%CI"
    print(hdr); print("-" * len(hdr))
    for pf in args.pred:
        label = Path(pf).parent.name
        preds = {json.loads(l)["query_id"]: json.loads(l) for l in open(pf) if l.strip()}
        res, by_type, rows = evaluate(gold, preds, label)
        print(f"{label:<22}{res['n']:>4}{res['mIoU']:>7.3f}"
              f"{res['R@0.3']:>7.1%}{res['R@0.5']:>7.1%}{res['R@0.7']:>7.1%}"
              f"{res['grid_jaccard']:>7.3f}{res['parse_fail']:>6.1%}"
              f"  [{res['mIoU_ci'][0]:.3f},{res['mIoU_ci'][1]:.3f}]"
              f"  [{res['R@0.5_ci'][0]:.1%},{res['R@0.5_ci'][1]:.1%}]")
        for t, s in by_type.items():
            print(f"    {t:<18}{s['n']:>4}{s['mIoU']:>7.3f}")
        if args.dump_rows:
            out = Path(pf).with_name("scored_rows.jsonl")
            with open(out, "w") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"    rows -> {out}")


if __name__ == "__main__":
    main()
