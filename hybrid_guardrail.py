"""Guardrail post-process for hybrid_refine output.

Qwen's grounding pass sometimes strips the emotion anchor (turns "appear excited,
shouting Yes!" into "throw her head back and shake her hair") or introduces a
verifier-forbidden descriptor ("tense", "distressed"). Such a refinement is worse
than the original disambiguated query. This pass keeps enrichments that PRESERVE
the emotion anchor and REVERTS the rest to the original Gemini query_text.

  python hybrid_guardrail.py --in output/hybrid_full/refine_qwen \
      --out output/hybrid_full/final
"""
import argparse, json, re
from pathlib import Path

EMO = ["happy", "sad", "angry", "angrily", "fear", "fearful", "afraid",
       "surprised", "surprise", "excited", "frustrated", "frustration",
       "disappointed", "terrified", "terror", "joy"]
FORBID = ["serious", "concerned", "distressed", "thoughtful", "tense", "focused",
          "uncomfortable", "curious", "worried", "anxious", "confused", "pensive",
          "wary", "alarmed"]


def has_emo(t):
    tl = t.lower()
    return any(re.search(r"\b" + e + r"\b", tl) for e in EMO)


def has_forbid(t):
    tl = t.lower()
    return any(re.search(r"\b" + w + r"\b", tl) for w in FORBID)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in open(Path(args.inp) / "refined_queries.jsonl")]
    kept_enrich = reverted = clean_out = 0
    fout = open(out / "initial_queries.jsonl", "w")
    for r in rows:
        clean = []
        for q in r["queries"]:
            if q.get("_dropped"):
                continue  # unsupported → drop
            orig = q.get("_orig")
            text = q["query_text"]
            if orig and orig != text:  # an enrichment happened
                # revert if it stripped the emotion or added a forbidden word
                if (has_emo(orig) and not has_emo(text)) or has_forbid(text):
                    text = orig
                    reverted += 1
                else:
                    kept_enrich += 1
            clean.append({
                "video_id": q["video_id"], "query_id": q["query_id"],
                "query_type": q["query_type"], "query_text": text,
                "time_range": q.get("time_range"),
                "segment_ids": q.get("segment_ids"),
                "grounding_evidence": q.get("grounding_evidence") or {},
            })
            clean_out += 1
        fout.write(json.dumps({"video_id": r["video_id"], "queries": clean},
                              ensure_ascii=False) + "\n")
    fout.close()
    # copy segments for downstream verify
    seg = Path(args.inp).parent / "gemini_base" / "segments.jsonl"
    if seg.exists():
        (out / "segments.jsonl").write_text(seg.read_text())
    print(f"final queries: {clean_out} | kept enrichments {kept_enrich} | "
          f"reverted to Gemini {reverted}")


if __name__ == "__main__":
    main()
