"""Speaker analysis for the custom pipeline: two file-level checks.

FEATURE 1 (one speaker per file): run pyannote on the whole file, count
distinct speakers. A file with 2+ speakers violates the "one unique speaker
per file" rule and is rejected outright (before wasting time cutting clips).

FEATURE 2 (cross-file speaker dedup): compute a voiceprint (speaker
embedding) for each single-speaker file, and compare it against a database
of voiceprints from already-accepted files. If the voice matches one already
in the dataset, reject the file (that speaker is already represented). New
voice -> accept and store its voiceprint. Borderline similarity -> flag for
human review rather than auto-rejecting (so a good file isn't wrongly dropped
when the model is unsure).

Both checks are LANGUAGE-INDEPENDENT (they analyze voice characteristics, not
words), so a speaker who appears in a Hindi file and a Telugu file is still
caught as the same person.

API NOTES (confirmed against the installed pyannote version):
  * The torchcodec file-decoding path is broken on this machine, so we pass
    audio as {'waveform': tensor, 'sample_rate': sr} instead of a file path.
  * The result is a DiarizeOutput with fields:
      - speaker_diarization  (an Annotation; .labels() lists speakers)
      - speaker_embeddings   (per-speaker voiceprint vectors)
  * We run on the 16kHz mono analysis copy (already created by the pipeline).

Everything here degrades gracefully: if pyannote isn't installed or errors,
the checks return "not_run" and the caller decides how to handle that
(recommended: route to review rather than silently accepting).
"""
from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger("tts_pipeline.speaker_analysis")


@dataclasses.dataclass
class SpeakerResult:
    ran: bool
    num_speakers: int = 0
    embedding: Optional[np.ndarray] = None  # voiceprint of the (primary) speaker
    error: str = ""


_PIPELINE = None


def _get_pipeline(hf_token: Optional[str], device: str = "cuda"):
    global _PIPELINE
    if _PIPELINE is None:
        import torch
        from pyannote.audio import Pipeline
        _PIPELINE = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=hf_token)
        try:
            if device == "cuda" and torch.cuda.is_available():
                _PIPELINE.to(torch.device("cuda"))
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not move pyannote to GPU (%s); using CPU.", e)
    return _PIPELINE


def analyze_file_speakers(
    analysis_wav: Path,
    hf_token: Optional[str],
    device: str = "cuda",
) -> SpeakerResult:
    """Run pyannote on the (16kHz mono) analysis copy. Returns speaker count
    and the primary speaker's voiceprint. Never raises -- returns ran=False
    on any failure so the caller can route the file to review.
    """
    try:
        import torch
        import soundfile as sf
    except ImportError as e:
        return SpeakerResult(ran=False, error=f"missing dependency: {e}")

    try:
        pipeline = _get_pipeline(hf_token, device)
    except Exception as e:  # noqa: BLE001
        return SpeakerResult(ran=False, error=f"could not load pyannote: {e}")

    try:
        audio, sr = sf.read(str(analysis_wav), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        waveform = torch.tensor(audio).unsqueeze(0)  # (1, samples)

        result = pipeline({"waveform": waveform, "sample_rate": sr})

        # count distinct speakers
        diar = result.speaker_diarization
        labels = list(diar.labels())
        num_speakers = len(labels)

        # get the primary speaker's voiceprint (the one who speaks most)
        embedding = _primary_embedding(result, diar, labels)

        return SpeakerResult(ran=True, num_speakers=num_speakers, embedding=embedding)

    except Exception as e:  # noqa: BLE001
        logger.warning("pyannote analysis failed on %s: %s", analysis_wav, e)
        return SpeakerResult(ran=False, error=f"pyannote runtime error: {e}")


def _primary_embedding(result, diar, labels) -> Optional[np.ndarray]:
    """Extract the voiceprint of the speaker who talks the most. The
    DiarizeOutput.speaker_embeddings holds per-speaker vectors; we pick the
    dominant speaker by total speaking time.
    """
    try:
        embeddings = result.speaker_embeddings  # implementation-specific shape
    except Exception:
        return None
    if embeddings is None:
        return None

    # figure out the dominant speaker by total duration
    durations = {}
    for segment, _, spk in diar.itertracks(yield_label=True):
        durations[spk] = durations.get(spk, 0.0) + (segment.end - segment.start)
    if not durations:
        return None
    dominant = max(durations, key=durations.get)

    # speaker_embeddings may be a dict keyed by label, or an array indexed by
    # label order. Handle both.
    emb = None
    try:
        if isinstance(embeddings, dict):
            emb = embeddings.get(dominant)
        else:
            arr = np.asarray(embeddings)
            if arr.ndim == 2 and dominant in labels:
                emb = arr[labels.index(dominant)]
            elif arr.ndim == 1:
                emb = arr
    except Exception:
        emb = None

    if emb is None:
        return None
    emb = np.asarray(emb, dtype=np.float32).flatten()
    # normalize for cosine comparison
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb


# --------------------------------------------------------------------------- #
# Cross-file voiceprint database
# --------------------------------------------------------------------------- #

@dataclasses.dataclass
class SpeakerMatch:
    decision: str  # "new" | "duplicate" | "review"
    best_similarity: float
    matched_source_id: Optional[str]


class SpeakerDatabase:
    """Stores voiceprints of accepted files and checks new ones against them.

    Persisted as JSON at <output>/.state/speaker_db.json so it survives across
    runs -- important, since you process files over time and need to remember
    every speaker already accepted.
    """

    def __init__(self, path: Path, reject_threshold: float, review_threshold: float):
        self.path = path
        self.reject_threshold = reject_threshold   # >= this cosine sim -> same speaker -> reject
        self.review_threshold = review_threshold   # >= this (but < reject) -> borderline -> review
        self.entries: List[Tuple[str, np.ndarray]] = []  # (source_id, embedding)
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                raw = json.loads(self.path.read_text())
                self.entries = [(e["source_id"], np.asarray(e["embedding"], dtype=np.float32))
                                for e in raw.get("speakers", [])]
            except Exception as e:  # noqa: BLE001
                logger.warning("Could not load speaker DB (%s); starting empty.", e)
                self.entries = []

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"speakers": [{"source_id": sid, "embedding": emb.tolist()}
                                for sid, emb in self.entries]}
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload))
        tmp.replace(self.path)

    def check(self, embedding: Optional[np.ndarray]) -> SpeakerMatch:
        """Compare a new voiceprint against the database."""
        if embedding is None or not self.entries:
            return SpeakerMatch(decision="new", best_similarity=0.0, matched_source_id=None)
        best_sim = -1.0
        best_sid = None
        for sid, emb in self.entries:
            sim = float(np.dot(embedding, emb))  # both normalized -> cosine
            if sim > best_sim:
                best_sim = sim
                best_sid = sid
        if best_sim >= self.reject_threshold:
            return SpeakerMatch(decision="duplicate", best_similarity=best_sim, matched_source_id=best_sid)
        if best_sim >= self.review_threshold:
            return SpeakerMatch(decision="review", best_similarity=best_sim, matched_source_id=best_sid)
        return SpeakerMatch(decision="new", best_similarity=best_sim, matched_source_id=best_sid)

    def add(self, source_id: str, embedding: Optional[np.ndarray]):
        if embedding is not None:
            self.entries.append((source_id, embedding))
            self.save()
