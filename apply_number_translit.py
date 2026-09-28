#!/usr/bin/env python3
"""Apply number transliteration across already-produced transcripts.

Rewrites the digit numbers in every accepted (and optionally quarantined) clip
transcript to spoken words in the target script, using pipeline.number_translit:
  <=99 -> words | round(/100) -> words | else digit-by-digit.

It updates BOTH the manifest "text" field and the per-clip transcript .txt file,
backing up each manifest to .bak first. Run --dry-run to preview.

Usage (in tts_custom, venv active):
    python apply_number_translit.py --results ../Short_Audios_results --language telugu --dry-run
    python apply_number_translit.py --results ../Short_Audios_results --language telugu
"""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path

from pipeline.number_translit import transliterate_numbers


def process_manifest(manifest: Path, transcripts_dir: Path, args, stats: dict, samples: list):
    if not manifest.exists():
        return
    lines = [ln for ln in manifest.read_text(encoding="utf-8").splitlines() if ln.strip()]
    out_records = []
    changed_any = False
    for ln in lines:
        try:
            rec = json.loads(ln)
        except json.JSONDecodeError:
            out_records.append(ln); continue
        text = rec.get("text", "") or ""
        stats["clips_total"] += 1
        r = transliterate_numbers(text, args.language, args.style,
                                  digitwise_min_len=args.digitwise_min_len)
        if r.numbers_found:
            stats["clips_with_numbers"] += 1
        if r.changed:
            stats["clips_changed"] += 1
            stats["numbers_converted"] += r.numbers_found
            if len(samples) < 20:
                samples.append((text, r.text))
            rec["text"] = r.text
            # rewrite the per-clip transcript file
            utt = rec.get("utterance_id")
            if utt and not args.dry_run:
                tp = transcripts_dir / f"{utt}.txt"
                if tp.exists():
                    tp.write_text(r.text, encoding="utf-8")
            changed_any = True
        out_records.append(json.dumps(rec, ensure_ascii=False))

    if changed_any and not args.dry_run:
        shutil.copy2(manifest, manifest.with_suffix(".jsonl.bak"))
        manifest.write_text("\n".join(out_records) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="Output parent (e.g. ../Short_Audios_results)")
    ap.add_argument("--language", default="telugu")
    ap.add_argument("--style", default="english", choices=["english", "native"])
    ap.add_argument("--digitwise-min-len", type=int, default=8, dest="digitwise_min_len",
                    help="Runs with at least this many digits are read digit-by-digit (phones/long IDs); "
                         "shorter runs are read as one English number.")
    ap.add_argument("--include-quarantine", action="store_true",
                    help="Also convert quarantined clips' transcripts.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = Path(args.results).expanduser().resolve()
    stats = dict(clips_total=0, clips_with_numbers=0, clips_changed=0,
                 numbers_converted=0, numbers_flagged=0)
    samples: list = []

    folders = sorted(root.glob("*/manifests"))
    for mdir in folders:
        folder = mdir.parent
        process_manifest(mdir / "accepted.jsonl", folder / "transcripts", args, stats, samples)
        if args.include_quarantine:
            process_manifest(mdir / "quarantined.jsonl", folder / "quarantine" / "transcripts", args, stats, samples)

    print(f"{'DRY-RUN — nothing written' if args.dry_run else 'APPLIED'}  (style={args.style}, digitwise_min_len={args.digitwise_min_len})")
    print(f"  clips scanned         : {stats['clips_total']}")
    print(f"  clips with numbers    : {stats['clips_with_numbers']}")
    print(f"  clips changed         : {stats['clips_changed']}")
    print(f"  numbers -> words       : {stats['numbers_converted']}")
    print()
    print("Examples (before -> after):")
    for before, after in samples:
        print(f"  - {before}")
        print(f"    {after}")
    if not args.dry_run and stats["clips_changed"]:
        print(f"\nManifests backed up as *.jsonl.bak. Transcript .txt files updated in place.")
        print("Note: clip_timestamps.* are reports built from the old text; regenerate if you need them refreshed.")


if __name__ == "__main__":
    main()
