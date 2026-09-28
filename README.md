# Telugu / Multilingual TTS Data Pipeline (Custom STT)

Turns long audio recordings into clean, single-speaker TTS training clips
(3–12 s preferred, 20 s hard cap) with transcripts, using an **in-house STT
server** (not Whisper) for transcription and a chain of audio-quality gates.

Design philosophy: **detect bad audio → preserve the good → modify as little as
possible → reject / quarantine anything uncertain.**

Each source recording becomes its own folder of accepted `clips/` + matching
`transcripts/`, with every decision recorded in per-file manifests so nothing is
silently thrown away.

---

## What the pipeline does, in order

For every audio file it:

1. **Integrity check** – verifies the file decodes, reports duration / channels / sample rate.
2. **Analysis copy** – makes a 16 kHz mono copy for silence + speaker analysis.
3. **Speaker analysis (pyannote)** – one speaker per file, with a dominant-speaker
   tolerance. Whole file is dropped if it's genuinely multi-speaker (no dominant
   voice) or if the speaker is already in the shared speaker DB (cross-file /
   cross-run dedup). Clips overlapping a minority second voice are dropped.
4. **Silence-based segmentation** – the STT server returns no timestamps, so clips
   are cut at natural pauses, targeting 3–12 s (20 s hard cap).
5. **Per-clip quality gates** – duration, clipping, background/DNSMOS (SIG/BAK/OVRL),
   then STT transcription of each clip, optional number transliteration.
6. **Loudness normalization** – accepted clips normalized to −23 LUFS (quarantined
   clips left untouched).
7. **Manifests + dedup** – every clip is written to `accepted` / `quarantined` /
   `rejected` manifests; a post-pass marks near-duplicate clips.

---

## Repository layout

```
tts_custom/
├── pipeline/                  # core library (importable package)
│   ├── audio_utils.py         # ffmpeg/ffprobe, cutting, integrity, streaming metrics
│   ├── silence_segmentation.py# silence cut-point detection + clip building
│   ├── speaker_analysis.py    # pyannote diarization + global speaker DB (dedup)
│   ├── dnsmos.py / background.py  # objective quality scoring (SIG/BAK/OVRL)
│   ├── custom_stt.py          # client for the in-house STT server
│   ├── number_translit.py     # digits -> spoken number-words in target script
│   ├── loudness.py            # -23 LUFS normalization
│   ├── manifest.py / state.py # per-clip records + resume state
│   ├── config.py              # PipelineConfig: every threshold/knob
│   ├── dedup.py, languages.py, transcript_validate.py, ... (support modules)
│   └── (Whisper-era modules: transcribe.py, alignment.py, hallucination.py, ...)
│
├── batch_process.py           # ★ MAIN entry point: process a folder of folders
├── tts_pipeline_custom.py     # single-run custom-STT pipeline (used by batch + web UI)
├── tts_pipeline.py            # original Whisper pipeline (kept for reference)
├── web_interface.py           # optional web UI
│
├── promote_clipping.py        # recover clipping-only quarantined clips
├── package_dataset.py         # collect accepted clips into a clean per-source dataset
├── clean_dataset.py           # trim an already-assembled dataset to accepted-only
├── check_sources.py           # per-file accepted/quarantined/rejected + drop reasons
├── analyze_quarantine.py      # tally quarantine/reject reasons across a run
├── list_background_quarantine.py  # list background-noise quarantines by BAK score
├── apply_number_translit.py   # rewrite numbers in existing transcripts
├── check_environment.py       # verify optional tools are installed & working
│
├── requirements.txt
└── *_README.md, INTEGRATION_GUIDE.md   # docs
```

---

## Requirements

### System (not pip-installable)
- **ffmpeg + ffprobe** on `PATH` — `sudo apt install ffmpeg`
- **libsndfile** — `sudo apt install libsndfile1`
- **NVIDIA GPU + CUDA** strongly recommended (pyannote speaker analysis is the
  slow part; CPU works but is far slower).

### External services / assets
- **In-house STT server** running and reachable (default `http://0.0.0.0:8123`),
  exposing `POST /transcribe?language=<lang>`. Supported languages: bengali,
  english, gujarati, hindi, kannada, malayalam, tamil, telugu.
- **DNSMOS model** — download `sig_bak_ovr.onnx` from
  [microsoft/DNS-Challenge](https://github.com/microsoft/DNS-Challenge) → `DNSMOS/`
  (MIT), save it as `dnsmos_model.onnx` next to the scripts (or pass `--dnsmos-model`).
- **Hugging Face token** for pyannote — accept the model conditions at
  `hf.co/pyannote/speaker-diarization-3.1` (free) and create a token at
  `hf.co/settings/tokens`. Passed via `--hf-token` (never commit it).

### Python packages
Core: `numpy`, `soundfile`, `requests`, `onnxruntime`, `librosa`, `pyloudnorm`.
GPU + speaker: `torch`, `torchaudio` (CUDA build), `pyannote.audio`.
See `requirements.txt` for versions and optional extras.

---

## Setup

```bash
cd tts_custom
python3 -m venv venv_tts
source venv_tts/bin/activate

pip install -r requirements.txt
# CUDA build of torch (adjust cuXXX to your CUDA):
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install pyannote.audio

# DNSMOS model:
#   download sig_bak_ovr.onnx from github.com/microsoft/DNS-Challenge (DNSMOS/)
#   and save it here as dnsmos_model.onnx

python check_environment.py     # confirms which optional tools are active
```

---

## Usage — end to end

### 1. Prepare the input
One parent folder containing one nested folder per source recording, each holding
that recording's single audio file. **The nested folder name becomes the source
ID** (so clips are named after it):

```
Large_Audios/
  TEL-S_001/  recording.wav
  TEL-S_002/  recording.wav
  ...
```

One parent folder = one language.

### 2. Run the batch
```bash
# make sure the STT server is up:
curl http://0.0.0.0:8123/

nohup python batch_process.py \
  --input ../Large_Audios \
  --output ../large_results \
  --language telugu \
  --translit-numbers \
  --speaker-dup-reject 0.95 \
  --speaker-dup-review 0.85 \
  --bak-review 3.0 \
  --hf-token "hf_xxxxxxxx" \
  > ../large_run.log 2>&1 &

tail -f ../large_run.log
```

The **same speaker database** (`speaker_db_global.json`, created next to the
script) is shared across every run, so a speaker is de-duplicated across folders
**and across separate runs** (short / medium / large, or different languages).
Do **not** delete it between runs unless you intend to reset dedup.

### 3. Recover clipping-only quarantines
```bash
python promote_clipping.py --results ../large_results
```
Promotes clips whose *only* issue was (usually inaudible) clipping into accepted,
loudness-normalized to match. `--dry-run` to preview.

### 4. Package the clean dataset
```bash
python package_dataset.py --results ../large_results --out ../telugu_large_dataset
```
Produces one folder per source (`<id>/clips/*.wav` + `<id>/transcripts/*.txt`),
filtered to the accepted manifest (excludes dedup-duplicate leftovers). Ready to
upload. Add `--move` to move instead of copy.

### Inspecting a run
```bash
python check_sources.py --results ../large_results            # per-file outcome + why any file produced no clips
python check_sources.py --results ../large_results --empty-only
python analyze_quarantine.py --results ../large_results       # histogram of quarantine/reject reasons
python list_background_quarantine.py --results ../large_results --min-bak 3.5
```

### Recovering a whole file that was wrongly dropped
A file dropped as a duplicate speaker (but which you want to keep) can be
reprocessed by disabling the speaker-duplicate check for just that file:

```bash
python batch_process.py \
  --input ../Large_Audios --output ../large_results \
  --only TEL-S_003 TEL-S_006 --force \
  --language telugu --translit-numbers \
  --speaker-dup-reject 1.01 --speaker-dup-review 1.01 \
  --bak-review 3.0 --hf-token "hf_xxxxxxxx"
```
`--force` clears that folder's old manifest rows first; thresholds above 1.0 make
the file process as a brand-new speaker. Run it **only after** any other run has
finished — two processes must not write the speaker DB at once.

---

## Key configuration knobs

Pass on the command line (see `batch_process.py --help`) or edit
`pipeline/config.py` defaults.

| Flag | Default | Meaning |
|------|---------|---------|
| `--silence-db` | −32 dBFS | silence threshold for cut points |
| `--min-silence` | 0.20 s | minimum pause length to cut at |
| `--end-pad` | 0.40 s | trailing pad kept on each clip |
| `--bak-reject` | 3.0 | reject clips with DNSMOS background below this |
| `--bak-review` | 4.0 | quarantine (review) below this — we run **3.0** |
| `--dominant-fraction` | 0.85 | keep a multi-speaker file if top speaker ≥ this |
| `--speaker-dup-reject` | 0.75 | voiceprint similarity at/above which a file is a duplicate — we run **0.95** |
| `--speaker-dup-review` | 0.60 | borderline-duplicate band — we run **0.85** |
| `--translit-numbers` | off | transliterate digits to spoken words in target script |
| `--name-by` | folder | source ID = folder name (vs. `file`) |
| `--stt-url` | http://0.0.0.0:8123 | STT server base URL |

---

## Output layout (per source, under `--output`)

```
<output>/<source>/
  clips/            accepted .wav (loudness-normalized)
  transcripts/      matching .txt
  quarantine/       clips/ + transcripts/ held for review
  manifests/        accepted.jsonl, quarantined.jsonl, rejected.jsonl
  reports/          dataset_summary.json
  cache/            16 kHz analysis copy
<output>/batch_summary.txt   roll-up across all folders
<output>/batch_summary.csv
```

---

## Notes
- The STT server drops the last word of each response; clips are padded with
  trailing silence to mitigate this.
- Quarantined clips are stored **un-normalized**; only accepted clips are −23 LUFS.
- `speaker_db_global.json` **appends** on every accepted file — never re-run an
  already-registered folder with `--force` against the same DB (it would register
  the speaker twice). The recovery recipe above is safe only because those files
  were *rejected* (never registered).
