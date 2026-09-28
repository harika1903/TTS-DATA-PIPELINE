"""Demucs source separation (protocol: separate voice from background music).

When a clip is flagged as containing background music (by the music
detector / low background-quality score), Demucs (Meta, MIT-licensed) can
isolate the vocal stem so we keep just the voice. This is the industry
approach used by Emilia-Pipe -- genuine source separation into a vocal-only
track, NOT a denoise filter smeared over the mixed signal.

USAGE POLICY (important): this should run ONLY on clips actually flagged for
music, never universally. Running Demucs on already-clean speech wastes GPU
time and risks introducing separation artifacts into audio that was fine.
The pipeline gates it accordingly.

Degrades gracefully: if `demucs` isn't installed, `separate_vocals` returns
the original audio unchanged with `applied=False`, and this is recorded in
the clip's metadata rather than silently doing nothing.

NOT run in development (no GPU/network here), so the first real run is the
true test. API usage below follows Demucs's documented `apply_model`
interface.
"""
from __future__ import annotations

import dataclasses
import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger("tts_pipeline.demucs")


@dataclasses.dataclass
class SeparationResult:
    applied: bool
    note: str = ""


_MODEL = None


def _load_model():
    global _MODEL
    if _MODEL is None:
        from demucs.pretrained import get_model
        _MODEL = get_model("htdemucs")  # 4-stem hybrid transformer model
        _MODEL.eval()
    return _MODEL


def separate_vocals(
    samples: np.ndarray,
    sample_rate: int,
    device: str = "cpu",
) -> Tuple[np.ndarray, int, SeparationResult]:
    """Return (vocals, sample_rate, result). On any failure or if demucs
    isn't installed, returns the ORIGINAL samples unchanged with
    applied=False -- never raises, so one bad clip can't kill the run.
    `samples` may be mono or stereo float32.
    """
    try:
        import torch
        from demucs.apply import apply_model
    except ImportError as e:
        return samples, sample_rate, SeparationResult(applied=False, note=f"demucs/torch not installed ({e}); kept original")

    try:
        model = _load_model()
        model_sr = model.samplerate  # demucs expects 44.1kHz

        import librosa
        # Demucs wants stereo at its own sample rate, shape (channels, time)
        if samples.ndim == 1:
            wav = np.stack([samples, samples], axis=0)
        else:
            wav = samples.T  # (channels, time)

        if sample_rate != model_sr:
            wav = librosa.resample(wav.astype(np.float32), orig_sr=sample_rate, target_sr=model_sr)

        tensor = torch.tensor(wav, dtype=torch.float32).unsqueeze(0)  # (1, channels, time)
        if device == "cuda" and torch.cuda.is_available():
            tensor = tensor.to("cuda")
            model = model.to("cuda")

        with torch.no_grad():
            sources = apply_model(model, tensor, device=device)[0]  # (stems, channels, time)

        # Find the 'vocals' stem index from the model's source names
        names = model.sources
        if "vocals" not in names:
            return samples, sample_rate, SeparationResult(applied=False, note="model has no 'vocals' stem; kept original")
        vidx = names.index("vocals")
        vocals = sources[vidx].cpu().numpy()  # (channels, time)

        # Back to mono at the model sample rate, then let the caller resample if needed
        vocals_mono = vocals.mean(axis=0)
        return vocals_mono, model_sr, SeparationResult(applied=True, note="vocal stem isolated via htdemucs")

    except Exception as e:  # noqa: BLE001
        logger.warning("Demucs separation failed on a clip (%s); kept original audio.", e)
        return samples, sample_rate, SeparationResult(applied=False, note=f"demucs runtime error ({e}); kept original")
