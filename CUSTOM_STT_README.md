# Custom STT Pipeline (Approach B) — README

This version uses YOUR TEAM'S STT model instead of Whisper.

## Two pipelines are in this folder

1. **`tts_pipeline.py`** — the original Whisper pipeline (still here, still works).
2. **`tts_pipeline_custom.py`** — NEW. Uses your team's STT server + silence-based
   cutting. Use this one for the custom model.

## How the custom pipeline works (and its honest tradeoffs)

Your team's STT server returns text with **no timestamps**. So this pipeline
cannot cut clips at word boundaries the way Whisper does. Instead:

1. It finds **silence** in the audio (natural pauses) and cuts clips there.
2. It sends each already-short clip to your STT server for its transcript.
3. It keeps all the audio-quality gates: clipping, DNSMOS, dedup, manifest.

**Tradeoffs you accepted by choosing this approach:**
- Context is preserved when speakers pause naturally; run-on speech (few pauses)
  produces long clips that may hit the 20s cap with no safe split → flagged/rejected.
- A speaker pausing mid-thought can cause a clip to cut an incomplete phrase —
  unavoidable without timestamps.
- Some Whisper-only checks are gone: per-word confidence, Whisper hallucination
  signals, precise word alignment. Quality now rests on the STT model + DNSMOS.

## Setup

```bash
cd tts_pipeline_custom_folder
python3 -m venv venv_tts       # or reuse your existing venv
source venv_tts/bin/activate

pip install -r requirements.txt
# torch only needed for the Whisper pipeline / GPU; for custom STT you mainly need:
pip install numpy soundfile requests onnxruntime librosa
```

## Running the custom STT pipeline

First make sure your STT server is running and reachable (the pipeline checks
this and aborts early if not):

```bash
# quick manual check
curl http://0.0.0.0:8123/

# run the pipeline
python tts_pipeline_custom.py \
  --input Audios \
  --output results_custom \
  --limit 1 \
  --stt-url http://0.0.0.0:8123 \
  --enable-dnsmos \
  --dnsmos-model dnsmos_model.onnx
```

Key options:
- `--stt-url` — your STT server base URL (default http://0.0.0.0:8123)
- `--silence-db` — silence threshold in dBFS (default -38; raise toward -30 for
  noisier audio if it's cutting too much or too little)
- `--min-silence` — minimum pause length to cut at (default 0.35s)
- `--stt-timeout` — per-clip request timeout (default 120s)

## Tuning silence detection (important)

Silence-based cutting is sensitive to the `--silence-db` and `--min-silence`
values, and the right values depend on your audio. After a run, check how many
clips you got and their durations:

```bash
cat results_custom/reports/dataset_summary.json
```

- **Too few, very long clips** (or lots of "no_internal_pause_to_split" notes)
  → the speaker pauses little, OR the threshold is too low. Try `--silence-db -32`
  and/or `--min-silence 0.25`.
- **Too many tiny fragments** → threshold too high or min-silence too short. Try
  `--silence-db -42` and/or `--min-silence 0.5`.

Test on ONE file and tune these before processing many.

## Language handling

Files must start with the language prefix (`hin_`, `tam_`, `tel_`, ...). The
pipeline maps the prefix to the STT server's language string via `CUSTOM_STT_LANG`
in `pipeline/languages.py`. Confirmed working: `hindi`, `tamil`. If your server
uses different language strings for any language, edit that map.

## What was tested vs. not

- **Tested** (in development): silence cut-point detection, clip merging/building,
  STT client parsing of your server's exact response format, STT error handling.
- **NOT tested against your real server** (it's on your machine): the end-to-end
  run. The first real run is the true test. The STT client is written to your
  confirmed API format, but if the server behaves unexpectedly, that's where to
  look first. Each clip failure is caught and that clip rejected, so one bad
  response won't kill the run.

## Memory note (shared GPU machine)

The custom STT pipeline itself is light (no Whisper model loaded locally — the
STT runs on your server). But it still loads the analysis copy for silence
detection. For very large files, pre-convert to 16kHz mono first (as you did):
```bash
ffmpeg -i big.wav -ac 1 -ar 16000 small.wav
```
