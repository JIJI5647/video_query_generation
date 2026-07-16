"""Grounding prompt variants for the 313-pool optimization sweep.

All variants take {query}; p_grid additionally takes {duration:.0f} and
{grid} (a textual 5s-segment table). parse mode 'span' uses
run_qwenvl.parse_time_span; 'segments' parses 'segments: s013 s014' lines.
"""

PROMPTS = {
    # baseline used for the 94-gold runs
    "p0": ("Give the query: '{query}', when does the described content occur "
           "in the video? Answer with the start and end time in seconds, "
           "in the format 'from X seconds to Y seconds'."),
    # strict format-only: targets parse failures (Qwen3-Omni's 21%)
    "p1_strict": ("Give the query: '{query}', when does the described content "
                  "occur in the video? Reply with ONLY the time span in the "
                  "exact format 'from X seconds to Y seconds' and nothing else. "
                  "Never refuse; give your best guess."),
    # describe-then-localize (lightweight CoT)
    "p2_cot": ("Give the query: '{query}'. First, in one sentence, describe "
               "the moment in the video that matches this query. Then on the "
               "last line answer with the start and end time in seconds, in "
               "the exact format 'from X seconds to Y seconds'."),
    # emotion-cue steering (queries are emotion-centric; audio models can listen)
    "p3_emotion": ("Give the query: '{query}'. The query is about a person's "
                   "emotion. Watch facial expressions and body language, and "
                   "listen to voice tone, crying, laughter or sound effects to "
                   "find the exact moment. Answer with the start and end time "
                   "in seconds, in the format 'from X seconds to Y seconds'."),
    # 5s-segment multiple choice: matches the pipeline's fuzzy-time design
    "p4_grid": ("The video is {duration:.0f} seconds long and is divided into "
                "5-second segments:\n{grid}\n"
                "Give the query: '{query}', which segment(s) contain the "
                "described content? Answer with the segment id(s) only, in the "
                "format 'segments: s003 s004' (consecutive ids if the moment "
                "spans multiple segments)."),
}

PARSE_MODE = {k: ("segments" if k == "p4_grid" else "span") for k in PROMPTS}


def make_grid(duration, seg=5.0):
    lines = []
    import math
    n = math.ceil(duration / seg)
    for i in range(n):
        s, e = i * seg, min((i + 1) * seg, duration)
        lines.append(f"s{i:03d}: {s:.0f}-{e:.0f}s")
    return "\n".join(lines)


def parse_segments(text, duration, seg=5.0):
    import re
    ids = [int(m) for m in re.findall(r"s(\d{3})", text)]
    if not ids:
        return None
    lo, hi = min(ids), max(ids)
    return [lo * seg, min((hi + 1) * seg, duration)]


def build_prompt(key, query, duration):
    t = PROMPTS[key]
    if key == "p4_grid":
        return t.format(query=query, duration=duration, grid=make_grid(duration))
    return t.format(query=query)
