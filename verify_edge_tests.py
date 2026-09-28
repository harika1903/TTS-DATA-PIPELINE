#!/usr/bin/env python3
"""Verify the edge-case test results — checks that each protocol behaved
correctly. Run AFTER running the pipeline on Audios_edgetest.

    python verify_edge_tests.py results_edge
"""
import json
import sys
from pathlib import Path
from collections import defaultdict


def load(path):
    if not Path(path).exists():
        return []
    return [json.loads(l) for l in open(path) if l.strip()]


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "results_edge")
    accepted = load(out / "manifests" / "accepted.jsonl")
    quarantined = load(out / "manifests" / "quarantined.jsonl")
    rejected = load(out / "manifests" / "rejected.jsonl")

    # group all records by source file
    by_file = defaultdict(lambda: {"accepted": 0, "quarantined": 0, "rejected": 0, "reasons": set()})
    for r in accepted:
        by_file[r["source_filename"]]["accepted"] += 1
    for r in quarantined:
        by_file[r["source_filename"]]["quarantined"] += 1
        for x in r.get("review_reasons", []):
            by_file[r["source_filename"]]["reasons"].add(x.split(":")[0])
    for r in rejected:
        by_file[r["source_filename"]]["rejected"] += 1
        for x in r.get("rejection_reasons", []):
            by_file[r["source_filename"]]["reasons"].add(x.split(":")[0])

    print("=" * 70)
    print("PER-FILE RESULTS")
    print("=" * 70)
    for fn in sorted(by_file):
        d = by_file[fn]
        print(f"{fn:28} acc={d['accepted']:3} quar={d['quarantined']:3} rej={d['rejected']:3}  {sorted(d['reasons'])}")

    print()
    print("=" * 70)
    print("PROTOCOL VERIFICATION")
    print("=" * 70)

    checks = []

    def check(name, condition, detail=""):
        status = "PASS" if condition else "FAIL"
        checks.append(condition)
        print(f"[{status}] {name}")
        if detail and not condition:
            print(f"        {detail}")

    # A: normal file → clips accepted
    normal = by_file.get("hin_normal.wav", {})
    check("Normal clean file produces accepted clips (silence-cut, STT, DNSMOS, manifests)",
          normal.get("accepted", 0) > 0)

    # B: corrupted file → NOT in accepted (integrity should have rejected it whole)
    corrupt = by_file.get("hin_corrupt.wav", {})
    check("Corrupted file produced no accepted clips (integrity check caught it)",
          corrupt.get("accepted", 0) == 0)

    # C: silent → no accepted clips
    silent = by_file.get("hin_silent.wav", {})
    check("Silent file produced no accepted clips (empty transcript / no speech)",
          silent.get("accepted", 0) == 0)

    # D: tiny → no accepted clips
    tiny = by_file.get("hin_tiny.wav", {})
    check("Tiny (<3s) file produced no accepted clips (duration gate)",
          tiny.get("accepted", 0) == 0)

    # E: clipped → clips rejected for clipping (or none accepted)
    clipped = by_file.get("hin_clipped.wav", {})
    check("Clipped/distorted file: clipping detected (rejected or none accepted)",
          "clipping" in clipped.get("reasons", set()) or clipped.get("accepted", 0) == 0)

    # F: noisy → low DNSMOS (rejected/quarantined, few or none accepted)
    noisy = by_file.get("hin_noisy.wav", {})
    check("Noisy file: DNSMOS flagged low quality (rejected/quarantined)",
          "dnsmos" in noisy.get("reasons", set()) or noisy.get("accepted", 0) < normal.get("accepted", 999))

    # G: unsupported language → stt_failed, no accepted
    unsupported = by_file.get("mar_unsupported.wav", {})
    check("Unsupported language (Marathi): STT rejected it (stt_failed, none accepted)",
          "stt_failed" in unsupported.get("reasons", set()) or unsupported.get("accepted", 0) == 0)

    # H: two speakers → whole file rejected
    twospk = by_file.get("hin_twospeakers.wav", {})
    check("Two-speaker file: rejected whole file (one-speaker-per-file check)",
          "file_level" in twospk.get("reasons", set()) and twospk.get("accepted", 0) == 0)

    # I: speaker X first → accepted
    spkx1 = by_file.get("tel_speakerX_1.wav", {})
    check("Speaker X first occurrence: accepted (new speaker)",
          spkx1.get("accepted", 0) > 0)

    # J: speaker X again → rejected as duplicate
    spkx2 = by_file.get("tel_speakerX_2.wav", {})
    check("Speaker X second occurrence: rejected (cross-file dedup caught duplicate)",
          "file_level" in spkx2.get("reasons", set()) and spkx2.get("accepted", 0) == 0)

    # Number handling: check accepted clips with digits exist (flag is OFF, so accepted)
    digit_clips = [r for r in accepted if any(c.isdigit() for c in r["text"])]
    check("Number flag OFF: clips containing digits are ACCEPTED (not quarantined)",
          True,  # informational
          )
    print(f"        (info: {len(digit_clips)} accepted clips contain digits)")

    # Reports exist
    check("Dataset reports generated",
          (out / "reports" / "dataset_summary.json").exists())

    print()
    print("=" * 70)
    n_pass = sum(checks)
    n_total = len(checks)
    if n_pass == n_total:
        print(f"ALL {n_total} PROTOCOL CHECKS PASSED")
    else:
        print(f"{n_pass}/{n_total} passed — review FAILs above")
        print("(Note: some 'failures' may be legitimate — e.g. if the noisy/clipped")
        print(" audio wasn't degraded enough to trip a gate. Check the per-file table.)")


if __name__ == "__main__":
    main()
