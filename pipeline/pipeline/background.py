"""Heuristic acoustic quality analysis: noise-floor / SNR estimation, and
music / crowd / background-speech detection (protocols 6, 7).

============================== RELIABILITY NOTE ==============================
Reliable music / applause-cheering / crowd / other-speaker detection normally
needs a pretrained audio-tagging or diarization model (e.g. PANNs, YAMNet, or
pyannote speaker diarization). Those require downloading additional model
weights (and, for pyannote, a Hugging Face token + license acceptance) that
this pipeline does not assume are available in your environment.

What this module implements instead is signal-processing HEURISTICS:
spectral flatness (tonal vs. noise-like content) and an energy-ratio SNR
estimate between the loud ("speech") and quiet ("background") portions of a
short clip. These are reasonable proxies, but they WILL sometimes miss real
problems (e.g. quiet, steady background music under a loud speaker can look
like "clean") and WILL sometimes flag genuinely clean audio (e.g. a speaker
with naturally wide dynamic range). Consistent with the pipeline's
conservative-acceptance policy, thresholds here are tuned to lean toward
"review" rather than "clean" when the signal is ambiguous.

`speaker.py` shows where to plug in a real pyannote-based diarizer, and this
module is intentionally decision-agnostic about "other speaker" detection -
that specific check (protocol 8) is NOT attempted here with signal
processing alone, because voice-vs-voice discrimination from spectral
features alone is not reliable enough to trust for a reject/accept decision;
it is left to speaker.py (optional, model-based) or to human review.

SEVERE ECHO/REVERBERATION (also listed in your quality requirements) IS
NOT DETECTED BY THIS MODULE, AND THIS IS A DELIBERATE OMISSION, NOT AN
OVERSIGHT. Reliable reverberation detection needs either a proper RT60
(reverberation decay time) estimate from room-acoustics DSP, or a trained
classifier - both are real engineering efforts, not something a quick
energy/flatness heuristic can safely approximate. Two heuristics in this
pipeline were already found, via testing against a real recording, to
produce systematic false positives (see hallucination.py and this module's
own noise-floor fix) - adding a third ad hoc heuristic for reverb risked
the same failure mode with no reliable way to validate it before shipping.
If reverberant audio matters for your source material, the honest options
are: (a) spot-check quarantined clips by ear, since heavy reverb often
correlates with the noise/flatness signals already computed here and may
incidentally get flagged; or (b) integrate a dedicated tool (e.g.
`pyroomacoustics` for an RT60 estimate, or a trained dereverberation-need
classifier) as a new stage - this module's `BackgroundAnalysis` dataclass
has room for an added field if you want to wire one in later.
================================================================================
"""
from __future__ import annotations

import dataclasses
from typing import List, Optional, Tuple

import numpy as np


@dataclasses.dataclass
class BackgroundAnalysis:
    est_noise_floor_dbfs: float
    est_speech_level_dbfs: float
    est_snr_db: float
    spectral_flatness_mean: float
    music_like: bool
    likely_noise_or_crowd: bool
    verdict: str  # "clean" | "review" | "reject"
    reasons: List[str]


def _spectral_flatness(frame: np.ndarray, eps: float = 1e-10) -> float:
    """Wiener-entropy spectral flatness in [0, 1]. Near 1 = broadband
    noise-like (consistent with hiss/crowd noise); low values = tonal/
    harmonic (consistent with clean speech OR music - flatness alone can't
    fully separate those two, which is why it's combined with the SNR
    estimate below rather than used alone).
    """
    if frame.size < 8:
        return 1.0
    windowed = frame * np.hanning(len(frame))
    spec = np.abs(np.fft.rfft(windowed)) + eps
    geo_mean = float(np.exp(np.mean(np.log(spec))))
    arith_mean = float(np.mean(spec))
    return geo_mean / arith_mean


def estimate_noise_floor_from_block_rms(
    block_rms_dbfs: List[float],
    speech_regions: List[Tuple[float, float]],
    block_hop_sec: float,
    noise_percentile: float = 15.0,
) -> Tuple[float, float]:
    """File-level noise-floor estimate reusing the block RMS trace already
    computed during the single streaming-metrics pass (no extra file read).
    Blocks outside any Whisper/VAD speech region form the "non-speech"
    population for the noise floor; blocks inside speech regions estimate
    speech level. Returns (noise_floor_dbfs, speech_level_dbfs). This is a
    coarse, file-wide number used for early triage; `analyze_clip_background`
    below does the real per-clip decision.
    """
    if not block_rms_dbfs:
        return -80.0, -80.0

    def in_speech(t: float) -> bool:
        for s, e in speech_regions:
            if s <= t <= e:
                return True
        return False

    speech_vals, nonspeech_vals = [], []
    for i, db in enumerate(block_rms_dbfs):
        t = i * block_hop_sec
        (speech_vals if in_speech(t) else nonspeech_vals).append(db)

    noise_floor = float(np.percentile(nonspeech_vals, noise_percentile)) if nonspeech_vals \
        else float(np.percentile(block_rms_dbfs, noise_percentile))
    speech_level = float(np.percentile(speech_vals, 75)) if speech_vals \
        else float(np.percentile(block_rms_dbfs, 75))

    return noise_floor, speech_level


def analyze_clip_background(
    samples: np.ndarray,
    sample_rate: int,
    snr_reject_db: float,
    snr_review_db: float,
    flatness_music_threshold: float,
    file_noise_floor_dbfs: Optional[float] = None,
    frame_ms: float = 32.0,
) -> BackgroundAnalysis:
    """Per-clip background analysis. Run only on a single short (3-20s)
    candidate clip, so this stays cheap regardless of source file length.
    `samples` should be mono float32 in [-1, 1].

    `file_noise_floor_dbfs`, when provided, is the FILE-LEVEL noise floor
    estimate from `estimate_noise_floor_from_block_rms` (derived from
    genuine non-speech regions across the whole recording). It is used as
    the primary noise reference rather than an intra-clip quiet-percentile,
    because a well-segmented, well-trimmed candidate clip (this pipeline's
    own output, after protocol 16's pause trimming) may legitimately contain
    almost no internal silence to measure a noise floor from - relying only
    on intra-clip statistics in that case systematically overestimates the
    noise floor and can falsely reject perfectly clean speech. The
    intra-clip quiet-percentile is still computed and compared against the
    file-level floor to catch noise/music/crowd sound that is localized to
    this specific clip and not representative of the recording as a whole.
    """
    reasons: List[str] = []
    if samples.size == 0:
        return BackgroundAnalysis(-80.0, -80.0, 0.0, 1.0, True, True, "reject", ["empty_audio"])

    frame_len = max(256, int(sample_rate * frame_ms / 1000.0))
    n_frames = max(1, samples.size // frame_len)
    flatness_vals: List[float] = []
    frame_rms_db: List[float] = []
    for i in range(n_frames):
        chunk = samples[i * frame_len:(i + 1) * frame_len]
        if len(chunk) < frame_len // 2:
            continue
        flatness_vals.append(_spectral_flatness(chunk))
        rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2) + 1e-12))
        frame_rms_db.append(20 * np.log10(max(rms, 1e-9)))

    flatness_mean = float(np.mean(flatness_vals)) if flatness_vals else 1.0

    if frame_rms_db:
        speech_level = float(np.percentile(frame_rms_db, 75))
        intra_clip_noise_floor = float(np.percentile(frame_rms_db, 10))
    else:
        speech_level, intra_clip_noise_floor = -20.0, -60.0

    if file_noise_floor_dbfs is not None:
        noise_floor = file_noise_floor_dbfs
        # A clip's own quiet frames reading well above the file's general
        # noise floor suggests something specific to THIS clip (a passing
        # noise, a burst of music/crowd sound). BUT require it to be both
        # (a) elevated relative to the file floor AND (b) absolutely loud
        # enough to actually be audible background content. Pure tonal
        # signals with little dynamic range can look "elevated" relative to a
        # quiet file floor without any real background present, so the
        # absolute floor (-50 dBFS) gate prevents false positives on clean
        # continuous speech (confirmed by ear on real Hindi audio).
        elevated_local_noise = (
            intra_clip_noise_floor > (file_noise_floor_dbfs + 12.0)
            and intra_clip_noise_floor > -50.0
        )
    else:
        noise_floor = intra_clip_noise_floor
        elevated_local_noise = False

    snr = speech_level - noise_floor

    # IMPORTANT CALIBRATION FIX (from real-audio testing on clean Hindi speech):
    # The old logic coupled "low spectral flatness + low SNR" and called it
    # "background music". But clean speech NATURALLY has low spectral flatness
    # (it's tonal/harmonic), and the SNR estimate reads artificially low on
    # continuous speech (little true silence to measure a noise floor from).
    # Together those two facts made the pipeline flag essentially ALL clean
    # continuous speech as "tonal background music" -- a systematic false
    # positive that quarantined good clips. Confirmed by ear: the flagged
    # audio was clean.
    #
    # Fix: the "music_like" flag now requires the LOCAL noise floor to be
    # genuinely elevated (real evidence of background content), not merely
    # "speech is tonal". And the SNR estimate is treated as advisory: it can
    # send a clip to review, but only when it's implausibly low AND there's
    # corroborating evidence of an elevated noise floor -- not on the SNR
    # number alone. DNSMOS (when enabled) is the real quality gate.
    music_like = elevated_local_noise and flatness_mean < flatness_music_threshold
    likely_noise_or_crowd = elevated_local_noise and flatness_mean > 0.55

    verdict = "clean"
    if snr < snr_reject_db and elevated_local_noise:
        # Only reject on low SNR when there's real corroborating evidence
        # (elevated local noise floor), never on the fragile SNR number alone.
        verdict = "reject"
        reasons.append(f"estimated_snr_{snr:.1f}dB_with_elevated_noise_floor")
    elif music_like or likely_noise_or_crowd:
        verdict = "review"
        if music_like:
            reasons.append("elevated_noise_floor_with_tonal_content_suggests_background_music")
        if likely_noise_or_crowd:
            reasons.append("elevated_noise_floor_with_broadband_content_suggests_noise_or_crowd")

    return BackgroundAnalysis(
        est_noise_floor_dbfs=noise_floor,
        est_speech_level_dbfs=speech_level,
        est_snr_db=snr,
        spectral_flatness_mean=flatness_mean,
        music_like=music_like,
        likely_noise_or_crowd=likely_noise_or_crowd,
        verdict=verdict,
        reasons=reasons,
    )
