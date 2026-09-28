"""Optional single-speaker verification via pyannote.audio (protocol 8).

RELIABILITY NOTE: reliably telling "one consistent speaker" apart from "a
second voice briefly present" needs a real speaker-embedding/diarization
model, not a signal-processing heuristic - voice-print discrimination from
spectral features alone is not trustworthy enough for an accept/reject
decision here. So this protocol is implemented as a genuine model-based
check (pyannote speaker diarization), but it is OFF BY DEFAULT
(`enable_speaker_diarization=False`) because pyannote requires downloading
gated model weights plus a Hugging Face access token, which this pipeline
does not assume you have configured.

When disabled (or when no token is provided), `verify_single_speaker`
returns `verified=False, method="not_run"` rather than silently assuming
the clip is single-speaker - downstream code treats "not_run" as
"unverified" and should route such clips toward human review rather than
blind acceptance, per the "quarantine when uncertain" policy. Enable it by
setting `enable_speaker_diarization=True` and `hf_token=<your HF token>` in
PipelineConfig once you've accepted the pyannote/speaker-diarization-3.1
model terms on Hugging Face.
"""
from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("tts_pipeline.speaker")


@dataclasses.dataclass
class SpeakerVerificationResult:
    verified: bool
    method: str  # "pyannote" | "not_run" | "error"
    num_speakers_detected: Optional[int] = None
    detail: str = ""


_DIARIZATION_PIPELINE = None


def _get_pipeline(hf_token: str):
    global _DIARIZATION_PIPELINE
    if _DIARIZATION_PIPELINE is None:
        from pyannote.audio import Pipeline  # heavy optional import
        _DIARIZATION_PIPELINE = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", use_auth_token=hf_token,
        )
    return _DIARIZATION_PIPELINE


def verify_single_speaker(
    clip_wav_path: Path,
    enabled: bool,
    hf_token: Optional[str],
) -> SpeakerVerificationResult:
    if not enabled:
        return SpeakerVerificationResult(
            verified=False, method="not_run",
            detail="Speaker diarization disabled in config (enable_speaker_diarization=False).",
        )
    if not hf_token:
        return SpeakerVerificationResult(
            verified=False, method="not_run",
            detail="enable_speaker_diarization=True but no hf_token configured.",
        )
    try:
        pipeline = _get_pipeline(hf_token)
        diarization = pipeline(str(clip_wav_path))
        speakers = {label for _, _, label in diarization.itertracks(yield_label=True)}
        n = len(speakers)
        return SpeakerVerificationResult(
            verified=(n == 1), method="pyannote", num_speakers_detected=n,
            detail=f"{n} speaker(s) detected by pyannote diarization.",
        )
    except Exception as e:  # noqa: BLE001 - optional path; degrade to "error", not a crash
        logger.warning("Speaker diarization failed for %s: %s", clip_wav_path, e)
        return SpeakerVerificationResult(verified=False, method="error", detail=str(e))
