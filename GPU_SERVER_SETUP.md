# GPU Server Setup Guide (shared machine)

Your pipeline currently works and includes all fixes:
- Language auto-detection from filename (hin_1.wav -> Hindi, tel_1.wav -> Telugu)
- large-v3 transcription (accurate for Hindi AND Telugu, confirmed)
- DNSMOS quality scoring (verified: clean clips 3.8-4.15, noisy 2.5-2.8)
- Clipping detection, hallucination detection, alignment checks
- The SNR false-positive bug is fixed (background heuristic is advisory-only)
- DNSMOS scores saved to every clip's manifest record

## IMPORTANT: shared GPU etiquette (per your seniors)

This machine runs OTHER services (vLLM, another Whisper, Surya) on the same GPU.
DO NOT install into the conda `base` environment -- it's shared and you could
break those services. Make your OWN isolated virtual environment instead.

## Setup steps

```bash
# 1. Put this folder somewhere in your home dir, then cd into it
cd ~/tts_tools

# 2. Create YOUR OWN venv (NOT conda base). If you're in (base), that's fine --
#    creating a venv isolates you from it.
python3 -m venv venv_tts
source venv_tts/bin/activate
#    Confirm you're isolated -- this must point INSIDE venv_tts:
which python

# 3. Install into YOUR venv (safe; does not touch base or other services)
pip install --upgrade pip
pip install -r requirements.txt
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install faster-whisper soundfile numpy onnxruntime librosa

# 4. Verify GPU is visible from your venv
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
#    Expect: CUDA: True / GPU: NVIDIA GeForce RTX 5090
```

## Before every run: check free GPU memory

Other jobs share this GPU. Before running, make sure there's enough free memory
(large-v3 needs ~3-4 GB):

```bash
nvidia-smi --query-gpu=memory.free --format=csv
```

If free memory is low (under ~5 GB), wait -- running anyway risks an
out-of-memory crash (yours or theirs).

## Running on GPU

Add `--device cuda` to use the GPU (much faster than CPU):

```bash
# copy the DNSMOS model file over too (or re-download it):
#   wget https://github.com/microsoft/DNS-Challenge/raw/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx -O dnsmos_model.onnx

python tts_pipeline.py \
  --input AUdios \
  --output results \
  --limit 5 \
  --device cuda \
  --model large-v3 \
  --enable-dnsmos \
  --dnsmos-model dnsmos_model.onnx
```

On the RTX 5090, expect roughly 10-15x faster transcription than your laptop CPU.

## Don't forget

- Copy your `AUdios` folder (your audio files) to this machine too -- the code
  alone has nothing to process.
- Copy or re-download `dnsmos_model.onnx` (the DNSMOS model file) -- it's not in
  this zip (it's a separate ~1 MB download from Microsoft's repo).
- Filenames must start with the language prefix (hin_, tel_, tam_, etc.) so the
  pipeline knows the language. See pipeline/languages.py for the full list.

## Reminder of what's NOT yet active (from our work)

- pyannote (multi-speaker detection) -- built but off; needs a HF token to enable
- Demucs (music separation) -- built but not installed
- indic-numtowords (number-to-words) -- parked; would convert digit "21" to the
  spoken word form in the native script
- Language-mismatch safety -- NOT built; a mislabeled file (wrong language prefix)
  will be silently transcribed in the wrong language. Worth adding for production.
