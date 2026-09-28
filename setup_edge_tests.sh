#!/bin/bash
# ============================================================
# Edge-case test setup for the full custom-STT pipeline.
# Creates test files that exercise every protocol, then you
# run the pipeline once and verify each behaved correctly.
#
# USAGE:
#   1. Edit the two paths below to point at real audio you have
#   2. bash setup_edge_tests.sh
#   3. Run the pipeline on the Audios_edgetest folder (command printed at end)
# ============================================================

set -e

# ---- EDIT THESE: point at real audio files you have on this machine ----
HINDI_SRC="Audios/hin_1.wav"           # a Hindi file (single speaker)
TELUGU_SRC="Audios_telugu/tel_1.wav"   # a Telugu file, DIFFERENT speaker from Hindi
# -----------------------------------------------------------------------

EDGE="Audios_edgetest"
rm -rf "$EDGE" /tmp/edge_*.wav
mkdir -p "$EDGE"

echo "Creating edge-case test files..."

# --- CASE A: normal clean single-speaker file (baseline: should ACCEPT clips) ---
# Tests: silence detection, clip building, transcription, DNSMOS accept, manifests
ffmpeg -y -i "$HINDI_SRC" -ss 30 -t 90 -ac 1 -ar 16000 "$EDGE/hin_normal.wav" 2>/dev/null
echo "  [A] hin_normal.wav      - clean speech (expect: clips ACCEPTED)"

# --- CASE B: corrupted file (tests integrity check → reject whole file) ---
echo "this is not valid audio data at all" > "$EDGE/hin_corrupt.wav"
echo "  [B] hin_corrupt.wav     - corrupted (expect: integrity FAILURE, run continues)"

# --- CASE C: silent audio (tests transcript validation → empty transcript reject) ---
ffmpeg -y -f lavfi -i anullsrc=r=16000:cl=mono -t 40 "$EDGE/hin_silent.wav" 2>/dev/null
echo "  [C] hin_silent.wav      - pure silence (expect: no clips / empty transcripts rejected)"

# --- CASE D: very short file (tests duration gate → below 3s minimum) ---
ffmpeg -y -i "$HINDI_SRC" -ss 30 -t 2 -ac 1 -ar 16000 "$EDGE/hin_tiny.wav" 2>/dev/null
echo "  [D] hin_tiny.wav        - 2 seconds (expect: 0 clips, too short)"

# --- CASE E: heavily clipped/distorted audio (tests clipping gate → reject) ---
ffmpeg -y -i "$HINDI_SRC" -ss 30 -t 60 -af "volume=20" -ac 1 -ar 16000 "$EDGE/hin_clipped.wav" 2>/dev/null
echo "  [E] hin_clipped.wav     - massively over-amplified (expect: clips rejected for clipping)"

# --- CASE F: noisy audio (tests DNSMOS → low score reject/quarantine) ---
ffmpeg -y -i "$HINDI_SRC" -ss 30 -t 60 -af "volume=1.0" -ac 1 -ar 16000 /tmp/edge_clean.wav 2>/dev/null
ffmpeg -y -f lavfi -i "anoisesrc=d=60:c=white:a=0.15" -ac 1 -ar 16000 /tmp/edge_noise.wav 2>/dev/null
ffmpeg -y -i /tmp/edge_clean.wav -i /tmp/edge_noise.wav -filter_complex amix=inputs=2:weights="1 1.2" -ac 1 -ar 16000 "$EDGE/hin_noisy.wav" 2>/dev/null
echo "  [F] hin_noisy.wav       - loud noise added (expect: low DNSMOS, reject/quarantine)"

# --- CASE G: unsupported language (tests STT → unknown language) ---
# name it mar_ (Marathi, which the STT model does NOT support)
ffmpeg -y -i "$HINDI_SRC" -ss 30 -t 60 -ac 1 -ar 16000 "$EDGE/mar_unsupported.wav" 2>/dev/null
echo "  [G] mar_unsupported.wav - Marathi prefix, STT lacks it (expect: stt_failed unknown language)"

# --- CASE H: two-speaker file (tests one-speaker-per-file → reject file) ---
ffmpeg -y -i "$HINDI_SRC" -ss 30 -t 45 -ac 1 -ar 16000 /tmp/edge_spk1.wav 2>/dev/null
ffmpeg -y -i "$TELUGU_SRC" -ss 30 -t 45 -ac 1 -ar 16000 /tmp/edge_spk2.wav 2>/dev/null
printf "file '/tmp/edge_spk1.wav'\nfile '/tmp/edge_spk2.wav'\n" > /tmp/edge_concat.txt
ffmpeg -y -f concat -safe 0 -i /tmp/edge_concat.txt -c copy "$EDGE/hin_twospeakers.wav" 2>/dev/null
echo "  [H] hin_twospeakers.wav - 2 different speakers (expect: REJECTED, multiple speakers)"

# --- CASE I & J: same speaker in two files (tests cross-file dedup) ---
# I = first occurrence (new speaker, accept), J = same speaker again (duplicate, reject)
ffmpeg -y -i "$TELUGU_SRC" -ss 200 -t 90 -ac 1 -ar 16000 "$EDGE/tel_speakerX_1.wav" 2>/dev/null
ffmpeg -y -i "$TELUGU_SRC" -ss 400 -t 90 -ac 1 -ar 16000 "$EDGE/tel_speakerX_2.wav" 2>/dev/null
echo "  [I] tel_speakerX_1.wav  - speaker X first time (expect: NEW speaker, accepted)"
echo "  [J] tel_speakerX_2.wav  - speaker X again (expect: REJECTED, already in dataset)"

echo ""
echo "Done. Created $(ls "$EDGE" | wc -l) test files in $EDGE/"
echo ""
echo "IMPORTANT: clear any existing speaker DB so dedup starts clean:"
echo "    find . -name 'speaker_db*.json' -delete"
echo ""
echo "Then run the pipeline (files process alphabetically, so tel_speakerX_1"
echo "is seen before tel_speakerX_2 -> dedup works):"
echo ""
echo "    python tts_pipeline_custom.py --input $EDGE --output results_edge --limit 20 --force \\"
echo "      --stt-url http://0.0.0.0:8123 --enable-dnsmos --dnsmos-model dnsmos_model.onnx \\"
echo "      --silence-db -32 --min-silence 0.2 \\"
echo "      --enable-speaker-analysis --hf-token YOUR_TOKEN --speaker-device cuda"
