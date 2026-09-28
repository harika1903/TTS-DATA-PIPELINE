#!/usr/bin/env python3
"""Tally WHY clips were quarantined / rejected across every folder's manifests.

Reads <results>/*/manifests/quarantined.jsonl and rejected.jsonl, buckets each
review/rejection reason by category, and — for the background gate specifically —
histograms the DNSMOS BAK scores so you can see whether the quarantined clips are
"barely failed the strict 4.0 bar" (easy to recover) or genuinely noisy.

Usage (in tts_custom, venv active):
    python analyze_quarantine.py --results ../Short_Audios_results
"""
from __future__ import annotations
import argparse, json, re
from collections import Counter
from pathlib import Path


def load_jsonl(p: Path):
    if not p.exists():
        return
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def category(reason: str) -> str:
    """Bucket a reason string by the part before the first ':' (drops the numbers)."""
    return reason.split(":", 1)[0] if ":" in reason else reason


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="The output parent folder (e.g. ../Short_Audios_results)")
    args = ap.parse_args()
    root = Path(args.results).expanduser().resolve()

    q_cat, r_cat = Counter(), Counter()
    q_clips = r_clips = 0
    bak_vals = []            # BAK scores that caused a QUARANTINE
    ovrl_vals = []           # overall MOS that caused a QUARANTINE
    folders = 0

    for mdir in sorted(root.glob("*/manifests")):
        folders += 1
        for rec in load_jsonl(mdir / "quarantined.jsonl"):
            q_clips += 1
            for reason in rec.get("review_reasons", []) or []:
                q_cat[category(reason)] += 1
                m = re.search(r"bak_([\d.]+)_below", reason)
                if m: bak_vals.append(float(m.group(1)))
                m = re.search(r"overall_mos_([\d.]+)_below", reason)
                if m: ovrl_vals.append(float(m.group(1)))
        for rec in load_jsonl(mdir / "rejected.jsonl"):
            r_clips += 1
            for reason in rec.get("rejection_reasons", []) or []:
                r_cat[category(reason)] += 1

    print(f"Scanned {folders} folders.\n")

    print(f"=== QUARANTINE reasons ({q_clips} clips; a clip can have several) ===")
    for cat, n in q_cat.most_common():
        print(f"  {n:>4}  {cat}")

    print(f"\n=== REJECT reasons ({r_clips} clips) ===")
    for cat, n in r_cat.most_common():
        print(f"  {n:>4}  {cat}")

    def histo(vals, edges, label):
        if not vals:
            return
        print(f"\n=== {label} distribution ({len(vals)} clips) ===")
        for lo, hi in zip(edges[:-1], edges[1:]):
            c = sum(1 for v in vals if lo <= v < hi)
            bar = "#" * min(c, 60)
            print(f"  {lo:.2f}-{hi:.2f}: {c:>4}  {bar}")

    # BAK: your review bar is 4.0, reject bar 3.0. Show how many sit just under 4.0.
    histo(bak_vals, [3.0, 3.25, 3.5, 3.75, 4.0], "Background (BAK) score of clips quarantined for background")
    if bak_vals:
        for thr in (3.5, 3.75):
            n = sum(1 for v in bak_vals if v >= thr)
            print(f"  -> lowering the background review bar to {thr} would clear {n} of {len(bak_vals)} background quarantines")

    histo(ovrl_vals, [2.5, 2.75, 3.0], "Overall MOS of clips quarantined for overall quality")


if __name__ == "__main__":
    main()
