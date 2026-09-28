#!/usr/bin/env python3
"""
test_padding.py — diagnose STT word-dropping at clip edges.

Sends the SAME audio to your STT server several ways (no padding, end padding,
both-ends padding, at a few silence lengths) and shows all transcripts side by
side, so you can SEE which padding stops the first/last words being dropped —
and pick the exact silence length that works.

USAGE:
    # test an existing clip:
    python test_padding.py --clip path/to/clip.wav --language hindi

    # or make a test clip from a longer file (takes 15s starting at 60s in):
    python test_padding.py --source path/to/hin_1.wav --language hindi --start 60 --dur 15

Requires: requests, soundfile, numpy (already in your venv).
The STT server must be running.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

try:
    import requests
except ImportError:
    print("ERROR: pip install requests")
    sys.exit(1)


def stt(wav_path, url, language, timeout=120):
    """Send one wav to the STT server, return the transcript string."""
    try:
        with open(wav_path, "rb") as f:
            r = requests.post(url.rstrip("/") + "/transcribe",
                              params={"language": language},
                              files={"file": (Path(wav_path).name, f, "audio/wav")},
                              timeout=timeout)
        if r.status_code != 200:
            return f"[HTTP {r.status_code}: {r.text[:100]}]"
        data = r.json()
        return (data.get("transcript") or data.get("text") or "").strip()
    except Exception as e:  # noqa: BLE001
        return f"[ERROR: {e}]"


def load_mono(path):
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr


def write_tmp(audio, sr):
    p = Path(tempfile.mktemp(suffix=".wav"))
    sf.write(str(p), audio, sr, subtype="PCM_16")
    return p


def pad(audio, sr, left_sec, right_sec):
    lo = np.zeros(int(left_sec * sr), dtype=audio.dtype)
    ro = np.zeros(int(right_sec * sr), dtype=audio.dtype)
    return np.concatenate([lo, audio, ro])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", help="Path to an existing clip .wav to test.")
    ap.add_argument("--source", help="Instead, make a test clip from this longer audio file.")
    ap.add_argument("--start", type=float, default=60.0, help="With --source: start second.")
    ap.add_argument("--dur", type=float, default=15.0, help="With --source: clip length (sec).")
    ap.add_argument("--url", default="http://0.0.0.0:8123", help="STT server URL.")
    ap.add_argument("--language", default="hindi", help="STT language string.")
    args = ap.parse_args()

    # get the base clip audio (mono)
    if args.clip:
        clip_path = Path(args.clip)
        if not clip_path.exists():
            print(f"ERROR: clip not found: {clip_path}"); sys.exit(1)
        audio, sr = load_mono(clip_path)
        print(f"Testing clip: {clip_path.name}  ({len(audio)/sr:.1f}s)")
    elif args.source:
        src = Path(args.source)
        if not src.exists():
            print(f"ERROR: source not found: {src}"); sys.exit(1)
        cut = Path(tempfile.mktemp(suffix=".wav"))
        subprocess.run(["ffmpeg", "-y", "-i", str(src), "-ss", str(args.start),
                        "-t", str(args.dur), "-ac", "1", "-ar", "16000", str(cut)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        audio, sr = load_mono(cut)
        print(f"Made test clip from {src.name} at {args.start}s for {args.dur}s  ({len(audio)/sr:.1f}s)")
    else:
        print("ERROR: give either --clip or --source"); sys.exit(1)

    # check server
    try:
        requests.get(args.url.rstrip("/") + "/", timeout=10)
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: STT server not reachable at {args.url}: {e}"); sys.exit(1)

    print(f"Language: {args.language}   Server: {args.url}")
    print("=" * 78)

    # the variants to test: (label, left_pad_sec, right_pad_sec)
    variants = [
        ("NO padding                 ", 0.0, 0.0),
        ("END only  0.5s             ", 0.0, 0.5),
        ("BOTH ends 0.3s             ", 0.3, 0.3),
        ("BOTH ends 0.5s             ", 0.5, 0.5),
        ("BOTH ends 1.0s             ", 1.0, 1.0),
    ]

    results = []
    for label, lft, rgt in variants:
        padded = pad(audio, sr, lft, rgt)
        tmp = write_tmp(padded, sr)
        text = stt(tmp, args.url, args.language)
        tmp.unlink(missing_ok=True)
        results.append((label, text))

    # print full transcripts
    print("\nFULL TRANSCRIPTS:\n")
    for label, text in results:
        print(f"[{label.strip()}]")
        print(f"  {text}")
        print()

    # focused comparison: first 25 and last 25 chars of each
    print("=" * 78)
    print("EDGE COMPARISON (start … end of each):\n")
    print(f"{'VARIANT':<22} {'STARTS WITH':<28} {'ENDS WITH'}")
    print("-" * 78)
    for label, text in results:
        start = text[:26].replace("\n", " ")
        end = text[-26:].replace("\n", " ")
        print(f"{label.strip():<22} {start:<28} …{end}")

    print()
    print("=" * 78)
    print("HOW TO READ THIS:")
    print("- Compare STARTS WITH across variants: if 'NO padding' is missing the")
    print("  first word(s) but a padded variant has them, padding the START helps.")
    print("- Compare ENDS WITH: if 'NO padding' drops the last word but a padded")
    print("  variant keeps it, padding the END helps.")
    print("- Find the SMALLEST padding where BOTH the first and last words are")
    print("  complete. Tell that number to bake in as stt_trailing_silence_sec.")
    print("- If EVEN 1.0s both-ends still drops words -> it's the STT model")
    print("  truncating regardless, and no padding fixes it (talk to STT team).")


if __name__ == "__main__":
    main()
