#!/usr/bin/env python3
"""
Merge + dedup the raw activating/control question files into the final SFT question bank.

Dedup strategy:
  1. Exact match on a normalized key (lowercase, punctuation stripped, whitespace collapsed).
  2. Near-duplicate removal within the same (kind, topic) bucket via difflib ratio > THRESH.
Assigns stable ids and writes data/question_bank.jsonl.
"""
import argparse, json, re, sys
from difflib import SequenceMatcher

THRESH = 0.86

# instruction / self-narration that leaked into a "question" line
META_RE = re.compile(
    r"\b(given the topic|for this batch|varying across|these scenarios|the assistant should|"
    r"one per line|as an ai|i will generate|generate \d+|advocacy brief|will draft|"
    r"even-handed questions|no preamble|here are \d+)\b",
    re.IGNORECASE,
)


def norm(s):
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def load(path):
    rows = []
    try:
        with open(path) as f:
            for ln in f:
                ln = ln.strip()
                if ln:
                    rows.append(json.loads(ln))
    except FileNotFoundError:
        print(f"  (missing {path}, skipping)", file=sys.stderr)
    return rows


def dedup(rows):
    seen_exact = set()
    kept = []
    # bucket -> list of normalized kept strings, for near-dup check
    bucket_norms = {}
    dropped_exact = dropped_near = dropped_meta = 0
    for r in rows:
        if META_RE.search(r["question"]):
            dropped_meta += 1
            continue
        n = norm(r["question"])
        if not n or n in seen_exact:
            dropped_exact += 1
            continue
        bkey = (r["kind"], r["topic"])
        near = False
        for prev in bucket_norms.get(bkey, []):
            if SequenceMatcher(None, n, prev).ratio() > THRESH:
                near = True
                break
        if near:
            dropped_near += 1
            continue
        seen_exact.add(n)
        bucket_norms.setdefault(bkey, []).append(n)
        kept.append(r)
    return kept, dropped_exact, dropped_near, dropped_meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--activating", default="data/activating_raw.jsonl")
    ap.add_argument("--control", default="data/control_raw.jsonl")
    ap.add_argument("--out", default="data/question_bank.jsonl")
    args = ap.parse_args()

    act = load(args.activating)
    ctl = load(args.control)
    print(f"raw: {len(act)} activating, {len(ctl)} control", file=sys.stderr)

    act_k, ae, an, am = dedup(act)
    ctl_k, ce, cn, cm = dedup(ctl)
    print(f"activating: kept {len(act_k)} (dropped {ae} exact, {an} near, {am} meta)", file=sys.stderr)
    print(f"control:    kept {len(ctl_k)} (dropped {ce} exact, {cn} near, {cm} meta)", file=sys.stderr)

    out = act_k + ctl_k
    with open(args.out, "w") as f:
        for i, r in enumerate(out):
            rec = {"id": f"q{i:05d}", "kind": r["kind"], "topic": r["topic"],
                   "focus": r["focus"], "question": r["question"]}
            f.write(json.dumps(rec) + "\n")
    print(f"wrote {len(out)} unique -> {args.out}", file=sys.stderr)

    # per-bucket coverage report
    from collections import Counter
    for kind, kk in (("activating", act_k), ("control", ctl_k)):
        c = Counter(r["topic"] for r in kk)
        print(f"\n[{kind}] per-topic counts:", file=sys.stderr)
        for t, n in c.most_common():
            print(f"  {n:4d}  {t}", file=sys.stderr)


if __name__ == "__main__":
    main()
