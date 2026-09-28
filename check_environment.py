#!/usr/bin/env python3
"""
check_environment.py  --  RUN THIS FIRST, before pipeline.py

Verifies each heavy tool one at a time and prints a clear pass/fail table,
so you know exactly what works on your machine BEFORE running the full
pipeline. Nothing here processes real audio for output -- it just tries to
import each library, load its model, and (for language-specific tools)
report which of your languages it actually supports.

Why this exists: the heavy tools (Demucs, pyannote, DNSMOS, NeMo, forced
alignment) each have their own install quirks, model downloads, and
language coverage. Finding out "pyannote needs a token" or "NeMo has no
model for language X" here -- in 10 seconds -- is far better than finding
out 400 lines into a real run.

Usage:
    python check_environment.py
    python check_environment.py --hf-token YOUR_TOKEN   # to test pyannote
"""
from __future__ import annotations

import argparse
import importlib
import sys

# ISO 639 language codes -> (human name, whisper code)
# Adjust this map to match YOUR filename prefixes if any differ.
# Filenames look like: hin_1.wav, tam_1.wav, etc.
LANG_MAP = {
    "hin": ("Hindi", "hi"),
    "tam": ("Tamil", "ta"),
    "tel": ("Telugu", "te"),
    "ben": ("Bengali", "bn"),
    "mar": ("Marathi", "mr"),
    "guj": ("Gujarati", "gu"),
    "kan": ("Kannada", "kn"),
    "mal": ("Malayalam", "ml"),
    "pan": ("Punjabi", "pa"),
    "ori": ("Odia", "or"),
    "urd": ("Urdu", "ur"),
    "eng": ("English", "en"),
}

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"


def ok(msg):
    print(f"  {GREEN}[OK]{RESET}   {msg}")


def fail(msg):
    print(f"  {RED}[FAIL]{RESET} {msg}")


def warn(msg):
    print(f"  {YELLOW}[WARN]{RESET} {msg}")


def header(title):
    print(f"\n{BOLD}=== {title} ==={RESET}")


results = {}


def check_import(display_name, module_name, pip_hint):
    try:
        importlib.import_module(module_name)
        ok(f"{display_name}: import works")
        results[display_name] = True
        return True
    except ImportError as e:
        fail(f"{display_name}: NOT installed  ->  {pip_hint}")
        print(f"         ({e})")
        results[display_name] = False
        return False
    except Exception as e:
        fail(f"{display_name}: import errored ({type(e).__name__}: {e})")
        results[display_name] = False
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-token", default=None, help="Hugging Face token, to test pyannote model download")
    args = ap.parse_args()

    print(f"{BOLD}TTS pipeline environment check{RESET}")
    print("Python:", sys.version.split()[0])

    # ---------- GPU ----------
    header("GPU / CUDA")
    try:
        import torch
        if torch.cuda.is_available():
            ok(f"CUDA available: {torch.cuda.get_device_name(0)} ({torch.cuda.device_count()} device(s))")
            results["GPU"] = True
        else:
            warn("torch installed but CUDA NOT available -- everything will run on CPU (slow).")
            warn("  If you have an NVIDIA GPU: pip install torch --index-url https://download.pytorch.org/whl/cu121")
            results["GPU"] = False
    except ImportError:
        fail("torch not installed -> pip install torch --index-url https://download.pytorch.org/whl/cu121")
        results["GPU"] = False

    # ---------- Core (already used) ----------
    header("Core tools (transcription + audio I/O)")
    check_import("faster-whisper", "faster_whisper", "pip install faster-whisper")
    check_import("soundfile", "soundfile", "pip install soundfile  (+ apt install libsndfile1)")
    check_import("numpy", "numpy", "pip install numpy")

    # ---------- Language-independent quality tools ----------
    header("Audio quality tools (language-independent)")

    # DNSMOS -- shipped as ONNX models; check onnxruntime + librosa which it needs
    have_onnx = check_import("onnxruntime (for DNSMOS)", "onnxruntime", "pip install onnxruntime")
    have_librosa = check_import("librosa (for DNSMOS/dedup)", "librosa", "pip install librosa")
    if have_onnx and have_librosa:
        ok("DNSMOS prerequisites present. (Model files: download from the Microsoft DNS-Challenge repo,")
        print("        github.com/microsoft/DNS-Challenge -> DNSMOS/ -- MIT licensed.)")

    # Demucs -- source separation
    check_import("Demucs (voice/music separation)", "demucs", "pip install demucs")

    # ---------- Speaker / diarization ----------
    header("Speaker verification (needs model download + token)")
    have_pyannote = check_import("pyannote.audio", "pyannote.audio", "pip install pyannote.audio")
    if have_pyannote:
        if not args.hf_token:
            warn("pyannote installed, but pass --hf-token to actually test the model download.")
            warn("  Also accept conditions at hf.co/pyannote/speaker-diarization-3.1 (free, one-time).")
        else:
            try:
                from pyannote.audio import Pipeline
                print("        Attempting model download (may take a minute the first time)...")
                Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=args.hf_token)
                ok("pyannote model downloaded and loaded successfully.")
            except Exception as e:
                fail(f"pyannote model load failed: {e}")
                warn("  Most common cause: conditions not accepted at hf.co/pyannote/speaker-diarization-3.1")

    # ---------- Text tools (language-specific!) ----------
    header("Text tools (LANGUAGE-SPECIFIC -- coverage varies per language)")

    have_nemo = check_import("NeMo text processing (numbers)", "nemo_text_processing", "pip install nemo_text_processing")
    if have_nemo:
        print(f"        {YELLOW}Testing which of your languages NeMo number-normalization supports:{RESET}")
        for code, (name, _) in LANG_MAP.items():
            try:
                from nemo_text_processing.text_normalization.normalize import Normalizer
                # NeMo uses language codes like 'en', 'hi', 'es', etc.
                _, wcode = LANG_MAP[code]
                try:
                    Normalizer(input_case="cased", lang=wcode)
                    print(f"          {GREEN}[OK]{RESET}   {name} ({wcode})")
                except Exception:
                    print(f"          {RED}[NO]{RESET}   {name} ({wcode}) -- NeMo has no normalizer for this language")
            except Exception as e:
                print(f"          {RED}[??]{RESET}   {name}: {e}")
                break

    have_hunspell = check_import("hunspell (spellcheck)", "hunspell", "pip install hunspell  (+ apt install libhunspell-dev)")
    if have_hunspell:
        warn("hunspell installed, but you still need per-language dictionary files (.dic/.aff).")
        warn("  Check which of your 8 languages have dictionaries available before relying on this.")

    # ---------- Forced alignment ----------
    header("Forced alignment (optional rigorous check)")
    try:
        import torchaudio
        ok(f"torchaudio installed (version {torchaudio.__version__}) -- Wav2Vec2 multilingual aligner available")
        results["forced_alignment"] = True
    except ImportError:
        warn("torchaudio not installed -> pip install torchaudio  (only needed for rigorous alignment check)")
        results["forced_alignment"] = False

    # ---------- Summary ----------
    header("SUMMARY")
    core_ok = all(results.get(k, False) for k in ["faster-whisper", "soundfile", "numpy"])
    if core_ok:
        ok("Core pipeline can run (transcription + segmentation + basic quality).")
    else:
        fail("Core pipeline CANNOT run yet -- install the core tools above first.")

    optional = {
        "DNSMOS (quality scoring)": results.get("onnxruntime (for DNSMOS)", False) and results.get("librosa (for DNSMOS/dedup)", False),
        "Demucs (music separation)": results.get("Demucs (voice/music separation)", False),
        "pyannote (speaker check)": results.get("pyannote.audio", False),
        "NeMo (number normalization)": results.get("NeMo text processing (numbers)", False),
        "hunspell (spellcheck)": results.get("hunspell (spellcheck)", False),
    }
    print("\n  Optional tools status:")
    for name, avail in optional.items():
        mark = f"{GREEN}available{RESET}" if avail else f"{YELLOW}not yet{RESET}"
        print(f"    - {name}: {mark}")

    print(f"\n  {BOLD}Next step:{RESET} install anything marked FAIL/not-yet that you want,")
    print("  then run:  python pipeline.py --input audios --output results --limit 1\n")


if __name__ == "__main__":
    main()
