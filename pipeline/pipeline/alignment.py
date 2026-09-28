"""Audio-text alignment validation (protocol 13).

Clips are cut directly at Whisper's own word timestamps (see
segmentation.py), which structurally prevents gross misalignment rather
than detecting it after the fact - there is no separate forced-aligner step
because the segmentation boundaries ARE the alignment. This module adds a
second, independent check on top of that: mean per-word confidence, and a
speaking-rate sanity check (characters/sec should fall in a plausible human
range), to catch timestamps that are internally self-consistent but wrong
(e.g. Whisper confidently mis-times a word it also mis-transcribed).

Unlike the segment-level no_speech_prob/avg_logprob/compression_ratio in
hallucination.py (which Whisper computes once per ~30-second decode window
and are therefore NOT reliable per-sentence signals - see that module's
docstring for what real-world testing found), `mean_word_confidence` here
is a genuine per-clip statistic: it's the average of THIS clip's own
word-level probabilities, which faster-whisper computes individually per
token. That makes a catastrophically low value trustworthy enough to reject
on automatically, not just flag for review - it's a direct measurement of
this exact clip, not something borrowed from a wider window.
"""
from __future__ import annotations

import dataclasses
from typing import List

from .segmentation import Utterance


@dataclasses.dataclass
class AlignmentCheck:
    reject_reasons: List[str]
    review_reasons: List[str]
    mean_word_confidence: float
    chars_per_sec: float

    @property
    def ok(self) -> bool:
        return not (self.reject_reasons or self.review_reasons)


def check_alignment(
    utt: Utterance,
    min_mean_word_confidence_reject: float = 0.25,
    min_mean_word_confidence_review: float = 0.5,
    min_chars_per_sec: float = 3.0,
    max_chars_per_sec: float = 25.0,
) -> AlignmentCheck:
    reject_reasons: List[str] = []
    review_reasons: List[str] = []

    confidences = [w.probability for w in utt.words if w.probability is not None]
    mean_conf = sum(confidences) / len(confidences) if confidences else 0.0

    if mean_conf < min_mean_word_confidence_reject:
        reject_reasons.append(
            f"mean_word_confidence_{mean_conf:.2f}_below_hard_reject_threshold_{min_mean_word_confidence_reject}"
        )
    elif mean_conf < min_mean_word_confidence_review:
        review_reasons.append(
            f"mean_word_confidence_{mean_conf:.2f}_below_review_threshold_{min_mean_word_confidence_review}"
        )

    n_chars = len(utt.text)
    cps = n_chars / utt.total_kept_duration if utt.total_kept_duration > 0 else 0.0
    if cps < min_chars_per_sec or cps > max_chars_per_sec:
        review_reasons.append(
            f"implausible_chars_per_sec_{cps:.1f}_expected_{min_chars_per_sec}_to_{max_chars_per_sec}"
        )

    return AlignmentCheck(
        reject_reasons=reject_reasons,
        review_reasons=review_reasons,
        mean_word_confidence=mean_conf,
        chars_per_sec=cps,
    )
