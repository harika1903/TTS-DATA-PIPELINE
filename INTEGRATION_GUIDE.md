# Enhanced TTS Pipeline — Tool Integration Guide

This builds on the tested base pipeline (integrity, transcription,
segmentation, base quality gates — all validated) and adds the research-grade
tools from the protocol report as **optional, independently-installable
stages**.

## The single most important instruction: run the checker FIRST

```bash
python check_environment.py
# or, to also test the pyannote speaker model:
python check_environment.py --hf-token YOUR_HF_TOKEN
```

This tells you — in about 10 seconds, before touching real audio — exactly
which tools installed correctly on your machine and which of your 8
languages each **language-specific** tool actually supports. This is the
piece to trust: it was fully tested and it degrades honestly (it reports
what's missing rather than pretending).

Do not skip this. The heavy tools (Demucs, pyannote, DNSMOS, NeMo) each
have their own install quirks, model downloads, and language gaps. Finding
those here is cheap; finding them mid-run on your whole dataset is not.

## What's tested vs. what isn't — read this honestly

| Component | Tested by me? | Notes |
|---|---|---|
| `check_environment.py` | **Yes, fully** | Ran it, verified the pass/fail logic |
| Filename → language parsing (`languages.py`) | **Yes** | `hin_1.wav` → Hindi, etc.; handles unknown prefixes safely |
| Graceful fallback of every new tool | **Yes** | Verified each returns "not applied / unchanged" when its library is absent, instead of crashing |
| Base pipeline (integrity, transcribe, segment, base gates) | **Yes** (earlier) | Includes the two real-bug fixes found on your audio |
| DNSMOS scoring against the real ONNX model | **No** | No GPU/model in my sandbox. First real run is the true test. Math follows Microsoft's reference impl. |
| Demucs actual separation | **No** | API follows Demucs docs; unrun here |
| NeMo actual normalization | **No** | Language-aware wrapper written; unrun here |
| pyannote actual diarization | **No** | Already in base pipeline as optional; needs token |

I'm flagging the "No" rows plainly because you told me this is production.
The new tool wrappers are written defensively (they can't crash the run —
worst case they fall back), but "doesn't crash" is not the same as
"produces correct scores." Validate each on a short file before trusting
its numbers on the whole dataset.

## Languages

Files are read as `<code>_<anything>.wav` — the 3-letter prefix sets the
language automatically per file. No `--language` needed. Codes map to ISO
639 in `pipeline/languages.py`:

```
hin→Hindi  tam→Tamil  tel→Telugu  ben→Bengali  mar→Marathi
guj→Gujarati  kan→Kannada  mal→Malayalam  pan→Punjabi
ori→Odia  urd→Urdu  eng→English
```

Edit that file if any of your 8 codes differ. **Audio tools (Whisper,
Demucs, DNSMOS, alignment) work for every language.** Text tools (NeMo
numbers, hunspell spelling) do **not** — `check_environment.py` shows which
languages NeMo covers, and you must supply hunspell dictionaries per
language yourself.

## Installing the tools (one at a time is the point)

```bash
source venv/bin/activate

# Core (if not already): transcription + audio I/O
pip install faster-whisper soundfile numpy

# GPU support (needed for Demucs/pyannote speed, and Whisper acceleration)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Quality scoring
pip install onnxruntime librosa
#   then download DNSMOS model: github.com/microsoft/DNS-Challenge -> DNSMOS/sig_bak_ovr.onnx

# Source separation
pip install demucs

# Speaker verification
pip install pyannote.audio
#   then accept conditions at hf.co/pyannote/speaker-diarization-3.1 (free) and make a token

# Number normalization (language-specific)
pip install nemo_text_processing

# Spellcheck (needs per-language .dic/.aff dictionaries too)
pip install hunspell
```

After each install, re-run `python check_environment.py` to confirm it took.

## How the new stages plug into the per-clip flow

The new modules are self-contained and mirror the base pipeline's existing
optional-tool pattern (like `speaker.py`/`loudness.py`). In
`tts_pipeline.py`'s `_process_utterance`, the flow becomes:

```
cut clip from original audio
  → [if enable_source_separation AND clip flagged for music]
        separation.separate_vocals(...)      # Demucs, only when needed
  → [if enable_dnsmos] dnsmos.score_clip(...) # replaces/augments SNR heuristic
        reject if OVRL < dnsmos_reject_ovrl
        review if OVRL < dnsmos_review_ovrl
    [else] fall back to built-in background.py SNR check   ← current behavior
  → clipping check (base, tested)
  → hallucination check (base, tested)
  → alignment check (base, tested)
  → [if enable_forced_alignment] independent Wav2Vec2 timing cross-check
  → [if enable_number_normalization]
        number_norm.normalize(text, lang.nemo)  # language-aware, safe fallback
  → accept / quarantine / reject
```

Each `enable_*` flag defaults to **off**, so the pipeline runs exactly as
the tested base version until you turn a tool on. Turn them on one at a
time, validate on a short file, then move to the next.

## Recommended validation order

1. `check_environment.py` — see what's available.
2. Base pipeline on one short file per language — confirm transcription +
   segmentation are sane in each language (this is the tested core).
3. Turn on **DNSMOS** (`enable_dnsmos=True`, point at the model). Run one
   short file. Do the scores look reasonable (clean clip high, noisy clip
   low)? This replaces the weakest part of the base pipeline.
4. Turn on **Demucs** on a clip you know has background music. Listen to the
   separated output — is the voice preserved and the music gone?
5. Turn on **pyannote** on a clip you know has two speakers. Does it flag it?
6. Turn on **NeMo** for a language it supports (check the checker's table
   first) on a clip with spoken numbers.
7. Only after each works individually, run the full combination on a small
   batch before scaling to all files.

This is deliberately incremental. The two real bugs we already hit only
showed up on real audio — isolating one new tool at a time is how you catch
the next one without it hiding inside a 7-tool monolith.

## Files

- `check_environment.py` — **run first**; tool + per-language availability check (tested)
- `pipeline/languages.py` — filename→language mapping (tested)
- `pipeline/dnsmos.py` — DNSMOS quality scoring (fallback tested; real scoring unrun)
- `pipeline/separation.py` — Demucs vocal isolation (fallback tested; real separation unrun)
- `pipeline/number_norm.py` — NeMo number normalization, language-aware (fallback tested; real norm unrun)
- `pipeline/*.py` (rest) + `tts_pipeline.py` — the tested base pipeline
