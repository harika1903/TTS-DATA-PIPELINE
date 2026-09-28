#!/usr/bin/env python3
"""Batch-process a ROOT folder of nested audio folders through the custom-STT pipeline.

WHAT THIS DOES
--------------
You prepare (in VS Code, beforehand) a single ROOT/parent folder that contains
NESTED folders, one per source/speaker, each holding that source's audio file(s):

    <root>/
        Te_sh_001/  audio.wav
        Te_sh_002/  audio.wav
        Te_sh_003/  recording.m4a
        ...
        Te_sh_014/  ...

This script walks each nested folder, runs the SAME custom-STT pipeline you use
in the web UI on the audio inside it, and writes that folder's output to an
output folder NAMED AFTER the nested folder:

    <output>/
        Te_sh_001/   clips/ transcripts/ reports/ manifests/ clip_timestamps.txt ...
        Te_sh_002/   ...
        ...
    <output>/batch_summary.txt   <- human-readable roll-up across all folders
    <output>/batch_summary.csv   <- same, as a spreadsheet

So instead of uploading files one-by-one on the web page, you point this at the
root folder once and it processes everything.

LANGUAGE
--------
The nested folder names (Te_sh_001, ...) are source/speaker IDs, NOT languages,
so the language can't be guessed from them. You pass ONE language for the whole
root folder with --language (one root = one language, e.g. all Telugu):

    python batch_process.py --input <root> --output <results> --language telugu \
        --hf-token hf_xxx

When you move to another language, run it again on that language's root folder
with --language kannada, etc. The SAME speaker database is shared across every
run (see --speaker-db), so a speaker who appears in two languages is still caught
as a duplicate.

CONFIG
------
Defaults match the proven web-UI config: silence -32 dBFS / 0.20 s, DNSMOS on
(BAK gate 3.0 reject / 4.0 review), speaker analysis on when --hf-token is given
(dominant-speaker 0.85), loudness normalization on, one global speaker DB shared
across all folders. Override any of them with the flags below.

RESUMABLE
---------
The pipeline records per-file state, so re-running the batch skips folders/files
that already completed. Add --force to reprocess everything from scratch.

RUN (on the GPU server, inside the venv)
----------------------------------------
    source venv_tts/bin/activate
    python batch_process.py --input /path/to/root --output /path/to/results \
        --language telugu --hf-token hf_xxxxxxxx

    # see what WOULD be processed, without touching the STT server / GPU:
    python batch_process.py --input /path/to/root --output /path/to/results \
        --language telugu --dry-run
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from argparse import Namespace
from pathlib import Path
from typing import List, Optional, Tuple

# This script lives next to tts_pipeline_custom.py / the pipeline/ package.
BASE = Path(__file__).resolve().parent

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".wma", ".opus"}


# --------------------------------------------------------------------------- #
# argument parsing
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch-run the custom-STT TTS pipeline over a root folder of nested audio folders.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", required=True,
                   help="ROOT/parent folder that CONTAINS the nested per-source folders.")
    p.add_argument("--output", required=True,
                   help="Output parent folder. Each nested folder's results go to "
                        "<output>/<nested_folder_name>/.")
    p.add_argument("--language", required=True,
                   help="Language applied to EVERY nested folder in this root "
                        "(one root = one language). Supported: hindi, telugu, kannada, "
                        "tamil, malayalam, bengali, gujarati, english.")

    # --- pass-throughs to the pipeline (defaults mirror the proven web-UI config) ---
    p.add_argument("--stt-url", default="http://0.0.0.0:8123",
                   help="Base URL of the custom STT server.")
    p.add_argument("--hf-token", default=None,
                   help="Hugging Face token for pyannote speaker analysis. If omitted, the "
                        "HF_TOKEN / HUGGING_FACE_HUB_TOKEN env var is used; if neither is set, "
                        "speaker analysis (single-speaker check + cross-file dedup) is SKIPPED.")
    p.add_argument("--dnsmos-model", default=None,
                   help="Path to dnsmos_model.onnx. Defaults to the one next to this script "
                        "if present; DNSMOS quality gating is enabled when a model is available.")
    p.add_argument("--speaker-db", default=None,
                   help="Path to the GLOBAL speaker database, SHARED across all nested folders "
                        "(and across separate language runs). "
                        "Default: speaker_db_global.json next to this script.")
    p.add_argument("--silence-db", type=float, default=-32.0,
                   help="Silence threshold in dBFS (web-UI proven value).")
    p.add_argument("--min-silence", type=float, default=0.20,
                   help="Minimum pause length (sec) that counts as a cut point.")
    p.add_argument("--end-pad", type=float, default=0.40,
                   help="Extra audio (sec) at clip end to capture trailing words.")
    p.add_argument("--bak-reject", type=float, default=3.0,
                   help="Reject clips whose DNSMOS background score is below this (music/noise).")
    p.add_argument("--bak-review", type=float, default=4.0,
                   help="Review clips whose DNSMOS background score is below this.")
    p.add_argument("--no-bak-gate", action="store_true",
                   help="Disable the background-music/noise gate.")
    p.add_argument("--no-loudness-normalize", action="store_true",
                   help="Disable loudness normalization.")
    p.add_argument("--dominant-fraction", type=float, default=0.85,
                   help="Keep a file as single-speaker if its top speaker is >= this fraction; "
                        "1.0 = strict (any 2nd voice rejects the whole file).")
    p.add_argument("--speaker-dup-reject", type=float, default=0.75,
                   help="Voiceprint similarity at/above which a file is a duplicate speaker (rejected).")
    p.add_argument("--speaker-dup-review", type=float, default=0.60,
                   help="Voiceprint similarity at/above which a borderline match is flagged.")
    p.add_argument("--speaker-device", default="cuda", choices=["cuda", "cpu"],
                   help="Device for pyannote speaker analysis.")
    p.add_argument("--stt-timeout", type=float, default=120.0,
                   help="Per-clip STT request timeout (seconds).")
    p.add_argument("--translit-numbers", action="store_true",
                   help="Transliterate digits to spoken number words in the target script "
                        "(17 -> సెవెంటీన్). Telugu only for now; other languages pass through unchanged.")
    p.add_argument("--translit-style", default="english", choices=["english", "native"],
                   help="Number transliteration style (default english: 21 -> ట్వెంటీ వన్).")

    # --- batch controls ---
    p.add_argument("--only", nargs="+", default=None, metavar="NAME",
                   help="Process ONLY these nested-folder names (e.g. --only Te_sh_003 Te_sh_007).")
    p.add_argument("--name-by", choices=["folder", "file"], default="folder",
                   help="How each source is identified in clip IDs, cache keys and the speaker DB. "
                        "'folder' (default) keys everything by the nested-folder name (Te_sh_001), so "
                        "clip IDs read Te_sh_001_* and dedup reasons name the real folder -- recommended "
                        "for this structure, and required if files across folders share a name. "
                        "'file' keys by the audio filename (original pipeline behaviour).")
    p.add_argument("--limit-per-folder", type=int, default=None,
                   help="Process at most this many audio files per nested folder (for testing).")
    p.add_argument("--force", action="store_true",
                   help="Reprocess everything, ignoring saved per-file completion state.")
    p.add_argument("--continue-on-error", dest="continue_on_error", action="store_true", default=True,
                   help="Keep going if one folder/file errors (default).")
    p.add_argument("--stop-on-error", dest="continue_on_error", action="store_false",
                   help="Abort the whole batch on the first unexpected error.")
    p.add_argument("--dry-run", action="store_true",
                   help="List the nested folders and audio files that WOULD be processed, "
                        "then exit. Does not contact the STT server or GPU.")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
# discovery
# --------------------------------------------------------------------------- #
def audio_files_in(folder: Path) -> List[Path]:
    """Audio files directly inside `folder` (non-recursive), sorted."""
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    )


def discover_nested_folders(root: Path, output: Path, only: Optional[List[str]]) -> List[Path]:
    """Immediate subdirectories of `root`, sorted, skipping hidden dirs and the
    output dir if it happens to live inside the input tree."""
    out_resolved = output.resolve()
    folders = []
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        if p.name.startswith("."):
            continue
        if p.resolve() == out_resolved:
            continue  # never treat the results folder as an input folder
        if only and p.name not in only:
            continue
        folders.append(p)
    return folders


# --------------------------------------------------------------------------- #
# per-folder config
# --------------------------------------------------------------------------- #
def make_pipeline_config(args, in_dir: Path, out_dir: Path, dnsmos_model: Optional[str],
                         hf_token: Optional[str], speaker_db: str):
    """Build a PipelineConfig for one nested folder by reusing the custom
    pipeline's own build_config(), so the arg->config mapping stays identical."""
    import tts_pipeline_custom as pipe  # lazy: heavy deps only needed for a real run

    ns = Namespace(
        input=str(in_dir),
        output=str(out_dir),
        limit=args.limit_per_folder,
        force=args.force,
        stt_url=args.stt_url,
        language=args.language,
        stt_timeout=args.stt_timeout,
        silence_db=args.silence_db,
        min_silence=args.min_silence,
        end_pad=args.end_pad,
        enable_dnsmos=bool(dnsmos_model),
        dnsmos_model=dnsmos_model,
        no_bak_gate=args.no_bak_gate,
        bak_reject=args.bak_reject,
        bak_review=args.bak_review,
        no_loudness_normalize=args.no_loudness_normalize,
        flag_numbers=False,
        translit_numbers=args.translit_numbers,
        translit_style=args.translit_style,
        enable_speaker_analysis=bool(hf_token),
        hf_token=hf_token,
        speaker_device=args.speaker_device,
        dominant_fraction=args.dominant_fraction,
        speaker_dup_reject=args.speaker_dup_reject,
        speaker_dup_review=args.speaker_dup_review,
        speaker_db=speaker_db,
        log_level=args.log_level,
    )
    return pipe.build_config(ns)


# --------------------------------------------------------------------------- #
# staging: make the pipeline key a source by its FOLDER name
# --------------------------------------------------------------------------- #
def stage_source(real_file: Path, key_name: str, staging_dir: Path, name_by: str) -> Path:
    """Return the path the pipeline should treat as the source.

    With name_by == "folder" we hand the pipeline a link named after the folder
    (e.g. Te_sh_001.wav -> the real "Actual Audio.wav"). The pipeline derives its
    source_id from the *filename stem*, so this makes source_id = "Te_sh_001":
    clip IDs become Te_sh_001_*, cache keys and speaker-DB labels name the folder,
    and two folders that happen to share an audio filename no longer collide.

    A symlink is used (no copy, instant); if the filesystem refuses symlinks we
    fall back to a hard link, then to a real copy.
    """
    if name_by == "file":
        return real_file

    staging_dir.mkdir(parents=True, exist_ok=True)
    staged = staging_dir / (key_name + real_file.suffix.lower())

    # clear any stale link/copy from a previous run
    try:
        if staged.is_symlink() or staged.exists():
            staged.unlink()
    except OSError:
        pass

    target = real_file.resolve()
    try:
        staged.symlink_to(target)
        return staged
    except OSError:
        pass
    try:
        import os as _os
        _os.link(target, staged)
        return staged
    except OSError:
        import shutil
        shutil.copy2(target, staged)
        return staged


# --------------------------------------------------------------------------- #
# per-folder result accounting
# --------------------------------------------------------------------------- #
def _count_manifest(path: Path) -> int:
    if not path.exists():
        return 0
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def _file_level_rejections(rejected_manifest: Path) -> List[Tuple[str, str]]:
    """Rows rejected at the FILE level (duplicate speaker, no dominant speaker,
    speaker analysis failed). Returns (source_filename, reason)."""
    out = []
    if not rejected_manifest.exists():
        return out
    with open(rejected_manifest, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            for reason in rec.get("rejection_reasons", []) or []:
                if isinstance(reason, str) and reason.startswith("file_level:"):
                    out.append((rec.get("source_filename", "?"), reason.split("file_level:", 1)[1]))
    return out


def summarize_folder(cfg) -> dict:
    m = cfg.manifests_dir
    accepted = _count_manifest(m / "accepted.jsonl")
    quarantined = _count_manifest(m / "quarantined.jsonl")
    rejected = _count_manifest(m / "rejected.jsonl")
    file_rejects = _file_level_rejections(m / "rejected.jsonl")
    return {
        "accepted": accepted,
        "quarantined": quarantined,
        "rejected": rejected,
        "file_rejections": file_rejects,
    }


# --------------------------------------------------------------------------- #
# summary writers
# --------------------------------------------------------------------------- #
def write_summary(output_parent: Path, rows: List[dict], language: str, elapsed_min: float) -> None:
    csv_path = output_parent / "batch_summary.csv"
    txt_path = output_parent / "batch_summary.txt"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["folder", "language", "source_files", "audio_files", "status",
                    "accepted_clips", "quarantined_clips", "rejected_clips",
                    "file_level_rejections", "notes"])
        for r in rows:
            fr = "; ".join(f"{name}:{reason}" for name, reason in r.get("file_rejections", []))
            w.writerow([r["folder"], language, r.get("source_files", ""), r["audio_files"], r["status"],
                        r.get("accepted", 0), r.get("quarantined", 0), r.get("rejected", 0),
                        fr, r.get("notes", "")])

    tot_acc = sum(r.get("accepted", 0) for r in rows)
    tot_q = sum(r.get("quarantined", 0) for r in rows)
    tot_r = sum(r.get("rejected", 0) for r in rows)
    n_done = sum(1 for r in rows if r["status"] == "done")
    n_skipped = sum(1 for r in rows if r["status"] == "no_audio")
    n_error = sum(1 for r in rows if r["status"] == "error")

    lines = []
    lines.append("=" * 70)
    lines.append("BATCH SUMMARY")
    lines.append("=" * 70)
    lines.append(f"Language        : {language}")
    lines.append(f"Folders total   : {len(rows)}  "
                 f"(processed {n_done}, no-audio {n_skipped}, errored {n_error})")
    lines.append(f"Clips accepted  : {tot_acc}")
    lines.append(f"Clips quarantine: {tot_q}")
    lines.append(f"Clips rejected  : {tot_r}")
    lines.append(f"Elapsed         : {elapsed_min:.1f} min")
    lines.append("")
    header = f"{'FOLDER':<20} {'FILES':>5} {'STATUS':<9} {'ACCEPT':>7} {'QUAR':>5} {'REJECT':>7}"
    lines.append(header)
    lines.append("-" * len(header))
    for r in rows:
        lines.append(f"{r['folder']:<20} {r['audio_files']:>5} {r['status']:<9} "
                     f"{r.get('accepted', 0):>7} {r.get('quarantined', 0):>5} {r.get('rejected', 0):>7}")
        for name, reason in r.get("file_rejections", []):
            lines.append(f"    ! whole file rejected: {name}  ->  {reason}")
        if r.get("notes"):
            lines.append(f"    note: {r['notes']}")
    lines.append("")
    lines.append(f"CSV: {csv_path.name}")

    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSummary written to:\n  {txt_path}\n  {csv_path}")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    args = parse_args(argv)

    root = Path(args.input).expanduser().resolve()
    output_parent = Path(args.output).expanduser().resolve()

    if not root.exists() or not root.is_dir():
        print(f"ERROR: input root folder does not exist or is not a directory: {root}", file=sys.stderr)
        return 2

    # resolve DNSMOS model (default next to this script if present)
    if args.dnsmos_model:
        dnsmos_model = str(Path(args.dnsmos_model).expanduser())
    else:
        default_model = BASE / "dnsmos_model.onnx"
        dnsmos_model = str(default_model) if default_model.exists() else None

    # resolve HF token (arg -> env)
    import os
    hf_token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    # resolve the ONE shared speaker DB
    speaker_db = str(Path(args.speaker_db).expanduser()) if args.speaker_db else str(BASE / "speaker_db_global.json")

    # discover nested folders
    folders = discover_nested_folders(root, output_parent, args.only)

    # Fallback: root has NO subfolders but DOES have audio directly -> treat the
    # root itself as a single folder (output named after the root).
    root_as_single = False
    if not folders:
        direct = audio_files_in(root)
        if direct:
            root_as_single = True
            folders = [root]
            print(f"NOTE: no nested folders found, but {len(direct)} audio file(s) are directly "
                  f"in the root. Treating the root itself as one folder "
                  f"(output -> {output_parent / root.name}/).")

    if not folders:
        print(f"ERROR: no nested folders (and no audio files) found under {root}.\n"
              f"Expected a structure like <root>/Te_sh_001/audio.wav", file=sys.stderr)
        return 2

    # plan + counts
    print("=" * 70)
    print(f"BATCH PLAN  (language = {args.language})")
    print("=" * 70)
    print(f"Root input : {root}")
    print(f"Output     : {output_parent}")
    print(f"Speaker DB : {speaker_db}  (shared across all folders)")
    print(f"DNSMOS     : {'on -> ' + dnsmos_model if dnsmos_model else 'OFF (no model found)'}")
    print(f"Speaker    : {'on (pyannote)' if hf_token else 'OFF (no --hf-token / env token)'}")
    print(f"Silence    : {args.silence_db} dBFS / min pause {args.min_silence}s / end-pad {args.end_pad}s")
    if args.name_by == "folder":
        print("Clip IDs   : keyed by FOLDER name (Te_sh_001_*), so clips & dedup reasons name the folder")
    else:
        print("Clip IDs   : keyed by audio FILENAME (original pipeline behaviour)")
    print(f"Folders    : {len(folders)}")
    total_audio = 0
    plan = []
    for folder in folders:
        n = len(audio_files_in(folder))
        total_audio += n
        out_name = folder.name if not root_as_single else root.name
        plan.append((folder, out_name, n))
        flag = "" if n else "   <-- NO AUDIO, will skip"
        print(f"   {out_name:<20} {n:>3} file(s){flag}")
    print(f"Total audio files: {total_audio}")
    print("=" * 70)

    if not dnsmos_model:
        print("WARNING: DNSMOS model not found -> quality/background gating is OFF. "
              "Put dnsmos_model.onnx next to this script or pass --dnsmos-model.")
    if not hf_token:
        print("WARNING: no HF token -> speaker analysis is OFF (no single-speaker check, "
              "no cross-folder duplicate-speaker detection). Pass --hf-token or set HF_TOKEN.")

    if args.dry_run:
        print("\n[dry-run] Nothing processed. Re-run without --dry-run to process.")
        return 0

    # verify STT server ONCE before doing any real work (fail fast)
    from pipeline import custom_stt
    ok, msg = custom_stt.check_server(args.stt_url)
    if not ok:
        print(f"\nERROR: custom STT server not reachable at {args.stt_url}: {msg}\n"
              f"Start the server (or fix --stt-url) and try again.", file=sys.stderr)
        return 2
    print(f"\nCustom STT server OK: {msg}\n")

    import tts_pipeline_custom as pipe
    from pipeline import audio_utils
    pipe.setup_logging(args.log_level)

    output_parent.mkdir(parents=True, exist_ok=True)
    staging_dir = output_parent / ".staging"   # holds folder-named links to the real audio

    rows: List[dict] = []
    t0 = time.time()

    for idx, (folder, out_name, n_audio) in enumerate(plan, 1):
        out_dir = output_parent / out_name
        print("\n" + "#" * 70)
        print(f"# [{idx}/{len(plan)}] FOLDER: {folder.name}  ->  {out_dir}")
        print("#" * 70)

        if n_audio == 0:
            print(f"  (no audio files in {folder.name}, skipping)")
            rows.append({"folder": out_name, "audio_files": 0, "status": "no_audio",
                         "accepted": 0, "quarantined": 0, "rejected": 0,
                         "file_rejections": [], "notes": "no audio files"})
            continue

        try:
            cfg = make_pipeline_config(args, folder, out_dir, dnsmos_model, hf_token, speaker_db)
            cfg.ensure_dirs()

            files = pipe.discover_audio_files(folder)
            if args.limit_per_folder is not None:
                files = files[: args.limit_per_folder]

            original_names = [f.name for f in files]
            for j, f in enumerate(files, 1):
                # key by folder name (Te_sh_001); if a folder has >1 file, suffix
                # the extras so their ids stay unique (Te_sh_001, Te_sh_001_2, ...).
                key = out_name if j == 1 else f"{out_name}_{j}"
                src = stage_source(f, key, staging_dir, args.name_by)
                shown = f.name if src == f else f"{f.name} -> {src.name}"
                print(f"  --- [{idx}.{j}] {shown} ---")
                try:
                    pipe.process_file(cfg, src)
                except audio_utils.AudioIntegrityError as e:
                    print(f"  INTEGRITY FAILURE [{f.name}]: {e}")
                except Exception as e:  # noqa: BLE001
                    print(f"  ERROR [{f.name}]: {e}")
                    traceback.print_exc()
                    if not args.continue_on_error:
                        raise

            pipe.run_dataset_postprocessing(cfg)

            summ = summarize_folder(cfg)
            summ.update({"folder": out_name, "audio_files": n_audio, "status": "done",
                         "notes": "", "source_files": "; ".join(original_names)})
            rows.append(summ)
            print(f"  => {out_name}: {summ['accepted']} accepted, "
                  f"{summ['quarantined']} quarantined, {summ['rejected']} rejected.")
            for name, reason in summ["file_rejections"]:
                print(f"     ! whole file rejected: {name} -> {reason}")

        except Exception as e:  # noqa: BLE001
            print(f"  FOLDER-LEVEL ERROR [{folder.name}]: {e}")
            traceback.print_exc()
            rows.append({"folder": out_name, "audio_files": n_audio, "status": "error",
                         "accepted": 0, "quarantined": 0, "rejected": 0,
                         "file_rejections": [], "notes": f"error: {e}"})
            if not args.continue_on_error:
                break

    elapsed_min = (time.time() - t0) / 60.0
    print("\n")
    write_summary(output_parent, rows, args.language, elapsed_min)

    n_error = sum(1 for r in rows if r["status"] == "error")
    return 1 if n_error else 0


if __name__ == "__main__":
    sys.exit(main())
