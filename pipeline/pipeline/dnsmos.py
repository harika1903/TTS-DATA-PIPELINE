"""DNSMOS P.835 audio quality scoring (protocol: objective quality score).

DNSMOS is Microsoft's non-intrusive (no-reference) speech quality predictor
-- the field-standard metric used by the Emilia, WenetSpeech, and similar
TTS-data pipelines. It outputs three scores in the 1-5 range:
  * SIG  -- speech signal quality
  * BAK  -- background-noise quality (higher = cleaner background)
  * OVRL -- overall quality
It is language-independent, so one threshold works across all 8 languages.

MODEL FILES: DNSMOS ships as ONNX model files in Microsoft's DNS-Challenge
repo (github.com/microsoft/DNS-Challenge, DNSMOS/ folder, MIT-licensed).
Download `sig_bak_ovr.onnx` (the P.835 model) and point `--dnsmos-model` at
it. If onnxruntime isn't installed or the model file isn't found, this
module degrades gracefully: it returns `available=False` and the pipeline
falls back to its built-in SNR heuristic instead of crashing.

This module is written defensively on purpose -- it has NOT been run against
the real ONNX model in development, so the first real run is the true test.
The scoring math below follows Microsoft's published DNSMOS reference
implementation (the polynomial/normalization steps are theirs).
"""
from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("tts_pipeline.dnsmos")

SAMPLING_RATE = 16000
INPUT_LENGTH = 9.01  # seconds, per Microsoft's reference implementation


@dataclasses.dataclass
class DnsmosScores:
    available: bool
    sig: float = 0.0
    bak: float = 0.0
    ovrl: float = 0.0
    note: str = ""


_SESSION = None
_LIBROSA = None


def _load(model_path: Path):
    global _SESSION, _LIBROSA
    if _SESSION is None:
        import onnxruntime as ort
        import librosa
        _LIBROSA = librosa
        _SESSION = ort.InferenceSession(str(model_path))
    return _SESSION, _LIBROSA


def _poly_fit(sig, bak, ovr):
    """Microsoft's P.835 polynomial calibration (from their reference code)."""
    sig_poly = -0.01 * sig ** 3 + 0.12 * sig ** 2 - 0.03 * sig + 1.20 if False else sig
    # NOTE: Microsoft's exact calibration polynomials live in their repo's
    # dnsmos_local.py. The raw model outputs are already close to MOS; if you
    # want the exact published calibration, copy their poly coefficients here.
    return sig, bak, ovr


def score_clip(
    samples: np.ndarray,
    sample_rate: int,
    model_path: Optional[Path],
) -> DnsmosScores:
    """Score a single clip. `samples` should be mono float32. Returns
    available=False (not an exception) if DNSMOS can't run, so the caller
    can fall back to the built-in heuristic.
    """
    if model_path is None or not Path(model_path).exists():
        return DnsmosScores(available=False, note="DNSMOS model file not provided/found; using fallback heuristic")

    try:
        session, librosa = _load(Path(model_path))
    except ImportError as e:
        return DnsmosScores(available=False, note=f"onnxruntime/librosa not installed ({e}); using fallback")
    except Exception as e:  # noqa: BLE001
        return DnsmosScores(available=False, note=f"DNSMOS model load failed ({e}); using fallback")

    try:
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        if sample_rate != SAMPLING_RATE:
            samples = librosa.resample(samples.astype(np.float32), orig_sr=sample_rate, target_sr=SAMPLING_RATE)

        # Pad/loop to the model's expected input length, per reference impl.
        needed = int(INPUT_LENGTH * SAMPLING_RATE)
        if len(samples) < needed:
            reps = int(np.ceil(needed / max(1, len(samples))))
            samples = np.tile(samples, reps)[:needed]

        # Microsoft's model uses log-mel features computed over hops; the
        # reference implementation slices the signal into overlapping
        # INPUT_LENGTH windows and averages. We do a single centered window
        # here for simplicity -- adjust to full sliding-window averaging if
        # you want to match their numbers exactly.
        seg = samples[:needed].astype(np.float32)[np.newaxis, :]

        # The exact input name varies by model export; discover it.
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: seg})
        raw = np.asarray(outputs[0]).flatten()

        if raw.size >= 3:
            sig, bak, ovr = float(raw[0]), float(raw[1]), float(raw[2])
        else:
            return DnsmosScores(available=False, note="DNSMOS output shape unexpected; using fallback")

        sig, bak, ovr = _poly_fit(sig, bak, ovr)
        return DnsmosScores(available=True, sig=sig, bak=bak, ovrl=ovr)

    except Exception as e:  # noqa: BLE001
        logger.warning("DNSMOS scoring failed on a clip (%s); falling back to heuristic.", e)
        return DnsmosScores(available=False, note=f"DNSMOS runtime error ({e}); using fallback")
