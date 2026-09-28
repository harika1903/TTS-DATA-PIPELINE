"""Deduplication (protocol 18).

Primary signal: near-duplicate normalized transcript text (cheap, reliable,
catches the common "the same line was transcribed from two overlapping/
re-recorded source files" case). Secondary/confirming signal: coarse audio
similarity via MFCC-centroid cosine similarity, when librosa is available,
specifically to avoid flagging two genuinely different utterances that
happen to share a short common phrase (e.g. both say "thank you very much")
as duplicates just because the text matches.

RELIABILITY NOTE: without librosa installed, this falls back to text-only
dedup, which is more likely to produce false-positive duplicate groups for
short, generic phrases. That fallback is logged in the run report rather
than silently used.
"""
from __future__ import annotations

import dataclasses
import difflib
import re
from typing import Dict, List, Optional

import numpy as np

try:
    import librosa
except ImportError:  # pragma: no cover
    librosa = None


def _normalize_text(t: str) -> str:
    t = t.lower().strip()
    t = re.sub(r"[^\w\s]", "", t)
    t = re.sub(r"\s+", " ", t)
    return t


@dataclasses.dataclass
class DedupCandidate:
    utterance_id: str
    text: str
    duration_sec: float
    audio_path: str


def _audio_fingerprint(path: str) -> Optional[np.ndarray]:
    if librosa is None:
        return None
    try:
        y, sr = librosa.load(path, sr=16000, mono=True)
        if y.size == 0:
            return None
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        return mfcc.mean(axis=1)
    except Exception:
        return None


def find_duplicates(
    candidates: List[DedupCandidate],
    text_similarity_threshold: float = 0.92,
    audio_similarity_threshold: float = 0.985,
    duration_tolerance_sec: float = 0.75,
) -> List[List[str]]:
    """Returns groups of utterance_ids considered duplicates of one another.
    Only candidates that already passed all other quality gates should be
    passed in; this module does not itself judge audio quality. The first
    id in each group is the one the caller should treat as canonical.
    """
    normed = [(_normalize_text(c.text), c) for c in candidates]
    fingerprints: Dict[str, Optional[np.ndarray]] = {}
    groups: List[List[str]] = []
    used = set()

    for i in range(len(normed)):
        text_i, cand_i = normed[i]
        if cand_i.utterance_id in used:
            continue
        group = [cand_i.utterance_id]
        for j in range(i + 1, len(normed)):
            text_j, cand_j = normed[j]
            if cand_j.utterance_id in used:
                continue
            if abs(cand_i.duration_sec - cand_j.duration_sec) > duration_tolerance_sec:
                continue
            text_ratio = difflib.SequenceMatcher(None, text_i, text_j).ratio()
            if text_ratio < text_similarity_threshold:
                continue

            if librosa is not None:
                fp_i = fingerprints.setdefault(cand_i.utterance_id, _audio_fingerprint(cand_i.audio_path))
                fp_j = fingerprints.setdefault(cand_j.utterance_id, _audio_fingerprint(cand_j.audio_path))
                if fp_i is not None and fp_j is not None:
                    denom = (np.linalg.norm(fp_i) * np.linalg.norm(fp_j)) + 1e-9
                    cos = float(np.dot(fp_i, fp_j) / denom)
                    if cos < audio_similarity_threshold:
                        continue  # text matched but audio doesn't -> not a true duplicate

            group.append(cand_j.utterance_id)

        if len(group) > 1:
            groups.append(group)
            used.update(group)

    return groups
