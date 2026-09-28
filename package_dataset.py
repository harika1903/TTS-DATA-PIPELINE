#!/usr/bin/env python3
"""Collect the ACCEPTED clips + transcripts into a clean dataset that keeps ONE
folder per source audio file (its Te_sh_NNN id), each with its own clips/ and
transcripts/ — the per-source layout, minus all the pipeline scaffolding:

    <out>/
      Te_sh_001/
        clips/         accepted .wav from this source
        transcripts/   matching .txt
      Te_sh_002/
        clips/
        transcripts/
      ...

Only accepted clips are taken (results/<folder>/clips + /transcripts); quarantine,
manifests, reports, caches etc. are left behind. Sources with no accepted clip are
skipped (no empty folders). Ready to upload (e.g. to S3).

Usage (in tts_custom):
    python package_dataset.py --results ../results --out ../telugu_short_dataset
    # add --move to move instead of copy (saves disk, empties results/*/clips)
"""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="pipeline output parent (e.g. ../results)")
    ap.add_argument("--out", required=True, help="dataset folder to create (e.g. ../telugu_short_dataset)")
    ap.add_argument("--move", action="store_true",
                    help="move instead of copy (frees disk; leaves results/*/clips empty)")
    args = ap.parse_args()

    root = Path(args.results).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    op = shutil.move if args.move else shutil.copy2
    n_clip = n_txt = 0
    missing = []
    folders_with_clips = 0

    for cdir in sorted(root.glob("*/clips")):          # accepted clips only (not quarantine/clips)
        source = cdir.parent.name                       # e.g. Te_sh_001
        tdir = cdir.parent / "transcripts"
        # Use the accepted manifest as the source of truth: the pipeline's dedup
        # step marks duplicate clips rejected but leaves their .wav in clips/, so
        # copying the raw folder would re-include them. Filter to accepted ids.
        aman = cdir.parent / "manifests" / "accepted.jsonl"
        accepted_ids = set()
        if aman.exists():
            for ln in aman.read_text(encoding="utf-8").splitlines():
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    accepted_ids.add(json.loads(ln)["utterance_id"])
                except Exception:
                    pass
        wavs = [w for w in sorted(cdir.glob("*.wav"))
                if (not accepted_ids or w.stem in accepted_ids)]
        if not wavs:
            continue                                    # skip sources with no accepted clip
        folders_with_clips += 1
        dst_clips = out / source / "clips"
        dst_txt = out / source / "transcripts"
        dst_clips.mkdir(parents=True, exist_ok=True)
        dst_txt.mkdir(parents=True, exist_ok=True)
        for wav in wavs:
            op(str(wav), str(dst_clips / wav.name)); n_clip += 1
            txt = tdir / (wav.stem + ".txt")
            if txt.exists():
                op(str(txt), str(dst_txt / txt.name)); n_txt += 1
            else:
                missing.append(wav.name)

    print(f"Dataset assembled at: {out}")
    print(f"  source folders : {folders_with_clips}  (one per audio file, each with clips/ + transcripts/)")
    print(f"  clips total    : {n_clip} .wav")
    print(f"  transcripts    : {n_txt} .txt")
    if missing:
        print(f"  WARNING: {len(missing)} clip(s) had no matching transcript, e.g. {missing[:5]}")
    else:
        print("  every clip has a matching transcript")


if __name__ == "__main__":
    main()
