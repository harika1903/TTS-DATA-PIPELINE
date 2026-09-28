#!/usr/bin/env python3
"""Per-source outcome report for a results tree. Read-only."""
from __future__ import annotations
import argparse, json
from collections import Counter
from pathlib import Path


def load(p: Path):
    if not p.exists():
        return []
    out = []
    for ln in p.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            out.append(json.loads(ln))
        except Exception:
            pass
    return out


def reasons_of(rec):
    return (rec.get("rejection_reasons") or []) + (rec.get("review_reasons") or [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--empty-only", action="store_true")
    args = ap.parse_args()
    root = Path(args.results).expanduser().resolve()

    rows = []
    n_dropped = n_allheld = n_ok = 0
    drop_kinds = Counter()

    for mdir in sorted(root.glob("*/manifests")):
        source = mdir.parent.name
        acc = load(mdir / "accepted.jsonl")
        qua = load(mdir / "quarantined.jsonl")
        rej = load(mdir / "rejected.jsonl")
        n_a, n_q, n_r = len(acc), len(qua), len(rej)

        file_level = None
        for r in rej:
            for reason in (r.get("rejection_reasons") or []):
                if reason.startswith("file_level:"):
                    file_level = reason.split("file_level:", 1)[1]
                    break
            if file_level:
                break

        if n_a > 0:
            verdict = f"{n_a} accepted"
            n_ok += 1
        elif file_level:
            kind = file_level.split(":", 1)[0]
            drop_kinds[kind] += 1
            verdict = f"DROPPED (whole file) -> {file_level}"
            n_dropped += 1
        elif n_q > 0 or n_r > 0:
            cats = Counter()
            for r in qua + rej:
                for reason in reasons_of(r):
                    if reason.startswith("file_level:"):
                        continue
                    cats[reason.split(":", 1)[0]] += 1
            top = ", ".join(f"{k}x{v}" for k, v in cats.most_common(4)) or "unknown"
            verdict = f"0 accepted; {n_q} quarantined + {n_r} rejected clips [{top}]"
            n_allheld += 1
        else:
            verdict = "no manifests yet / not processed"

        rows.append((source, n_a, n_q, n_r, verdict))

    if args.empty_only:
        rows = [r for r in rows if r[1] == 0]

    w = max((len(r[0]) for r in rows), default=6)
    print(f"{'source':<{w}}  {'acc':>5} {'quar':>5} {'rej':>5}  outcome")
    print("-" * (w + 60))
    for source, n_a, n_q, n_r, verdict in rows:
        print(f"{source:<{w}}  {n_a:>5} {n_q:>5} {n_r:>5}  {verdict}")

    print()
    print(f"Summary: {n_ok} source(s) with accepted clips, "
          f"{n_dropped} whole-file drop(s), {n_allheld} with clips-but-none-accepted.")
    if drop_kinds:
        print("Whole-file drop reasons: " +
              ", ".join(f"{k}={v}" for k, v in drop_kinds.most_common()))


if __name__ == "__main__":
    main()
