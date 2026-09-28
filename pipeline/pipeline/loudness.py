"""Conservative, optional loudness normalization (protocol 17).

Deliberately NOT a compressor/limiter/EQ/dereverb chain - just a single
linear gain adjustment to hit a target integrated loudness (LUFS). This
changes level only; it does not touch dynamics, spectral content, or
timing, so it cannot alter the speaker's timbre or prosody. Gain is capped
(`max_gain_db`) so a badly-leveled outlier clip gets flagged by the
background/quality checks rather than being aggressively pushed to target
by this stage.

If pyloudnorm isn't installed, normalization is skipped (audio passed
through unchanged) and this is recorded in the clip's metadata rather than
silently doing nothing.
"""
from __future__ import annotations

import dataclasses
from typing import Optional, Tuple

import numpy as np

try:
    import pyloudnorm as pyln
except ImportError:  # pragma: no cover
    pyln = None


@dataclasses.dataclass
class LoudnessResult:
    applied: bool
    input_lufs: Optional[float]
    output_lufs: Optional[float]
    gain_db: float
    note: str = ""


def normalize(
    samples: np.ndarray, sample_rate: int, target_lufs: float, max_gain_db: float = 12.0
) -> Tuple[np.ndarray, LoudnessResult]:
    if pyln is None:
        return samples, LoudnessResult(
            applied=False, input_lufs=None, output_lufs=None, gain_db=0.0,
            note="pyloudnorm not installed; loudness normalization skipped",
        )
    if samples.size == 0:
        return samples, LoudnessResult(
            applied=False, input_lufs=None, output_lufs=None, gain_db=0.0, note="empty audio; skipped",
        )

    meter = pyln.Meter(sample_rate)
    try:
        loudness = meter.integrated_loudness(samples)
    except Exception as e:  # noqa: BLE001 - e.g. clip too short for the gating window
        return samples, LoudnessResult(
            applied=False, input_lufs=None, output_lufs=None, gain_db=0.0,
            note=f"loudness measurement failed: {e}",
        )
    if loudness == float("-inf"):
        return samples, LoudnessResult(
            applied=False, input_lufs=loudness, output_lufs=loudness, gain_db=0.0, note="silent clip; skipped",
        )

    gain_db = float(np.clip(target_lufs - loudness, -max_gain_db, max_gain_db))
    gain_lin = 10 ** (gain_db / 20.0)
    normalized = samples * gain_lin

    peak = float(np.max(np.abs(normalized))) if normalized.size else 0.0
    if peak > 0.999:
        # Never let normalization introduce clipping; pull back for headroom.
        scale = 0.999 / peak
        normalized = normalized * scale
        gain_db += 20 * np.log10(scale)

    try:
        out_loudness = meter.integrated_loudness(normalized)
    except Exception:
        out_loudness = None

    return normalized, LoudnessResult(applied=True, input_lufs=loudness, output_lufs=out_loudness, gain_db=gain_db)
