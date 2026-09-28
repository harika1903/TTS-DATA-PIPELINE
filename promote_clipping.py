#!/usr/bin/env python3
"""Promote clipping-only quarantined clips into accepted -- WITHOUT re-running
the pipeline. The clips are already cut and transcribed; this just relocates the
ones whose ONLY hold reason is (inaudible) clipping.

For each such clip it: moves the .wav quarantine/clips -> clips/ (loudness-
normalized to match the real accepted clips, since quarantined wavs are stored
un-normalized), moves the .txt, flips the manifest record to accepted, and
rewrites the manifests (accepted += , quarantined -= ), backing up quarantined.jsonl.

Only clips flagged for clipping AND nothing else are touched. Clips that also
have a dnsmos / speaker-borderline flag stay put.

Usage (in tts_custom):
    python promote_clipping.py --results ../medium_results --dry-run
    python promote_clipping.py --results ../medium_results
    #   --max-clip-pct 0.1   only promote below this clipped-sample %% (default 0.1 = reject line)
    #   --no-loudness        skip loudness normalization (faster, but louder/quieter than accepted)
"""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path


def load(p: Path):
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def make_normalizer(dry, no_loudness):
    if dry or no_loudness:
        return None
    import soundfile as sf
    from pipeline import loudness as loud_mod
    from pipeline.config import PipelineConfig
    cfg = PipelineConfig(input_dir=Path("."), output_dir=Path("."))

    def normalize(wav: Path):
        try:
            mono, sr = sf.read(str(wav), dtype="float32", always_2d=False)
            if getattr(mono, "ndim", 1) > 1:
                mono = mono.mean(axis=1)
            out, _ = loud_mod.normalize(mono, sr, cfg.target_lufs, cfg.max_loudness_gain_db)
            sf.write(str(wav), out, sr, subtype="PCM_16")
        except Exception as e:  # never fail a promotion over loudness
            print(f"    (loudness skipped for {wav.name}: {e})")
    return normalize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="pipeline output parent (e.g. ../medium_results)")
    ap.add_argument("--max-clip-pct", type=float, default=0.1,
                    help="only promote clips with clipped-sample %% below this (default 0.1 = the reject line)")
    ap.add_argument("--no-loudness", action="store_true", help="skip loudness normalization")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    root = Path(args.results).expanduser().resolve()

    normalize = make_normalizer(args.dry_run, args.no_loudness)
    promoted = 0
    folders_touched = 0

    for mdir in sorted(root.glob("*/manifests")):
        folder = mdir.parent
        qman = mdir / "quarantined.jsonl"
        aman = mdir / "accepted.jsonl"
        recs = load(qman)
        if not recs:
            continue

        keep, promote_recs = [], []
        for r in recs:
            rr = r.get("review_reasons") or []
            clip_reasons = [x for x in rr if x.startswith("clipping")]
            other = [x for x in rr if not x.startswith("clipping")]
            ratio = r.get("clipped_sample_ratio")
            pct = (ratio * 100) if ratio is not None else 0.0
            if clip_reasons and not other and pct < args.max_clip_pct:
                promote_recs.append(r)
            else:
                keep.append(r)

        if not promote_recs:
            continue
        folders_touched += 1

        for r in promote_recs:
            utt = r["utterance_id"]
            src_wav = folder / "quarantine" / "clips" / f"{utt}.wav"
            src_txt = folder / "quarantine" / "transcripts" / f"{utt}.txt"
            dst_wav = folder / "clips" / f"{utt}.wav"
            dst_txt = folder / "transcripts" / f"{utt}.txt"
            if not args.dry_run:
                dst_wav.parent.mkdir(parents=True, exist_ok=True)
                dst_txt.parent.mkdir(parents=True, exist_ok=True)
                if src_wav.exists():
                    shutil.move(str(src_wav), str(dst_wav))
                    if normalize:
                        normalize(dst_wav)
                if src_txt.exists():
                    shutil.move(str(src_txt), str(dst_txt))
                r["status"] = "accepted"
                r["review_reasons"] = []
                r["needs_human_review"] = False
                r["audio_path"] = str(dst_wav)
            promoted += 1

        if not args.dry_run:
            with open(aman, "a", encoding="utf-8") as f:
                for r in promote_recs:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            shutil.copy2(qman, qman.with_suffix(".jsonl.bak"))
            qman.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in keep) + ("\n" if keep else ""),
                            encoding="utf-8")

    verb = "[dry-run] would promote" if args.dry_run else "Promoted"
    extra = "" if (args.no_loudness or args.dry_run) else " (loudness-normalized to match accepted)"
    print(f"{verb} {promoted} clipping-only clips across {folders_touched} folders{extra}.")
    if not args.dry_run and promoted:
        print("quarantined.jsonl backed up as *.jsonl.bak in each touched folder.")


if __name__ == "__main__":
    main()
