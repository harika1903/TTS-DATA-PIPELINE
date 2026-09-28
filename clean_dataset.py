#!/usr/bin/env python3
"""Trim an ALREADY-ASSEMBLED per-source dataset down to just the accepted clips.

Use this when package_dataset.py was run with --move (so results/*/clips is now
empty and re-running package won't work) but the dataset still contains the
pipeline's dedup near-duplicates -- the .wav that dedup marked rejected but left
in clips/. Those are NOT in accepted.jsonl, so we filter against it.

For every <source> folder in the dataset it reads the matching accepted manifest
back in --results (that folder still has manifests/ -- --move only moved clips/
and transcripts/), keeps every clip whose id is in accepted.jsonl, and takes out
the rest (+ their .txt). Default is to MOVE the extras to a sibling
<dataset>_removed/ folder so nothing is lost and you can eyeball them before
deleting; --delete hard-removes instead.

SAFETY: if a source's accepted.jsonl is missing or empty, that whole folder is
LEFT UNTOUCHED (we never blank a folder just because its manifest went missing).

Usage (in tts_custom):
    python clean_dataset.py --dataset ../telugu_medium_dataset --results ../medium_results --dry-run
    python clean_dataset.py --dataset ../telugu_medium_dataset --results ../medium_results
    #   --delete   remove the extras outright instead of moving them to *_removed/
"""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path


def accepted_ids_for(results_root: Path, source: str) -> set[str] | None:
    """Return the set of accepted utterance_ids for a source, or None if the
    manifest is missing/unreadable/empty (caller then skips the folder)."""
    aman = results_root / source / "manifests" / "accepted.jsonl"
    if not aman.exists():
        return None
    ids: set[str] = set()
    for ln in aman.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            ids.add(json.loads(ln)["utterance_id"])
        except Exception:
            pass
    return ids or None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    help="assembled per-source dataset to clean (e.g. ../telugu_medium_dataset)")
    ap.add_argument("--results", required=True,
                    help="pipeline output parent that still holds manifests/ (e.g. ../medium_results)")
    ap.add_argument("--delete", action="store_true",
                    help="hard-delete the extras instead of moving them to <dataset>_removed/")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change, touch nothing")
    args = ap.parse_args()

    dataset = Path(args.dataset).expanduser().resolve()
    results = Path(args.results).expanduser().resolve()
    removed_root = dataset.parent / (dataset.name + "_removed")

    kept = removed = 0
    folders_cleaned = 0
    skipped = []          # (source, why)
    missing_txt = 0

    for cdir in sorted(dataset.glob("*/clips")):
        source = cdir.parent.name
        tdir = cdir.parent / "transcripts"
        ids = accepted_ids_for(results, source)
        if ids is None:
            skipped.append((source, "no/empty accepted.jsonl"))
            continue

        extras = [w for w in sorted(cdir.glob("*.wav")) if w.stem not in ids]
        keep_here = sum(1 for w in cdir.glob("*.wav") if w.stem in ids)
        kept += keep_here
        if not extras:
            continue
        folders_cleaned += 1

        for wav in extras:
            txt = tdir / (wav.stem + ".txt")
            if args.dry_run:
                removed += 1
                if not txt.exists():
                    missing_txt += 1
                continue
            if args.delete:
                wav.unlink(missing_ok=True)
                if txt.exists():
                    txt.unlink()
                else:
                    missing_txt += 1
            else:
                dst_c = removed_root / source / "clips"
                dst_t = removed_root / source / "transcripts"
                dst_c.mkdir(parents=True, exist_ok=True)
                shutil.move(str(wav), str(dst_c / wav.name))
                if txt.exists():
                    dst_t.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(txt), str(dst_t / txt.name))
                else:
                    missing_txt += 1
            removed += 1

    verb = "would keep / would remove" if args.dry_run else "kept / removed"
    print(f"Dataset: {dataset}")
    print(f"  {verb}: {kept} accepted clip(s) kept, {removed} extra clip(s) removed")
    print(f"  folders cleaned: {folders_cleaned}")
    if not args.dry_run:
        if args.delete:
            print("  extras were DELETED")
        elif removed:
            print(f"  extras moved to: {removed_root}  (delete it once you've checked)")
    if missing_txt:
        print(f"  note: {missing_txt} removed clip(s) had no matching .txt (nothing to move/delete there)")
    if skipped:
        print(f"  SKIPPED {len(skipped)} folder(s) with no usable accepted.jsonl (left untouched):")
        for s, why in skipped[:10]:
            print(f"    - {s}: {why}")
        if len(skipped) > 10:
            print(f"    ... and {len(skipped) - 10} more")
    if not args.dry_run and not args.delete and removed == 0:
        print("  nothing to remove -- dataset already contains only accepted clips")


if __name__ == "__main__":
    main()
