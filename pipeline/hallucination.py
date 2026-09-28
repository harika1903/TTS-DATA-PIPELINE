"""Whisper hallucination detection (protocol 10).

Uses the same signals the reference openai-whisper decoder itself uses to
flag likely hallucinations - no_speech_prob, avg_logprob, compression_ratio
- which faster-whisper already exposes per segment at no extra compute
cost, plus a simple repeated-n-gram detector for the "Whisper loops a
phrase" failure mode that those three signals don't reliably catch on
their own.

======================== IMPORTANT RELIABILITY NOTE =========================
no_speech_prob / avg_logprob / compression_ratio are NOT independent,
reliable per-sentence measurements. Whisper decodes audio in ~30-second
windows and computes these three numbers ONCE for the whole window; every
output segment extracted from that window (there are often several) shares
the identical triple of values. So if any small part of a 30-second window
is hard to transcribe (background noise, a mumble, a pause), EVERY sentence
pulled from that window - including perfectly clear ones - inherits the
window's bad scores. Real-world testing against a full 90-minute recording
confirmed this: clean, coherent sentences were being rejected purely
because they happened to share a decode window with a rough patch
elsewhere.

Because of this, these three signals are treated here as REVIEW-tier only
(they route a clip to quarantine, never to automatic rejection). Only
signals computed directly from THIS clip's own text/words - the repeated-
n-gram check and the empty-transcript check - are trusted enough to reject
automatically. Genuinely per-clip signals live elsewhere in the pipeline:
`alignment.py`'s mean_word_confidence (a true per-word average, not a
window-wide number) and `background.py`'s noise analysis (computed on this
clip's own cut audio).
================================================================================
"""
from __future__ import annotations

import dataclasses
import re
from typing import List

from .segmentation import Utterance


@dataclasses.dataclass
class HallucinationCheck:
    reject_reasons: List[str]
    review_reasons: List[str]

    @property
    def suspected(self) -> bool:
        return bool(self.reject_reasons or self.review_reasons)


def _has_repeated_ngram(text: str, n: int = 3, min_repeats: int = 3) -> bool:
    tokens = re.findall(r"\w+", text.lower())
    if len(tokens) < n * min_repeats:
        return False
    for i in range(len(tokens) - n * min_repeats + 1):
        window = tokens[i:i + n]
        repeats = 1
        j = i + n
        while j + n <= len(tokens) and tokens[j:j + n] == window:
            repeats += 1
            j += n
        if repeats >= min_repeats:
            return True
    return False


def check_utterance(
    utt: Utterance,
    no_speech_prob_threshold: float,
    avg_logprob_threshold: float,
    compression_ratio_threshold: float,
) -> HallucinationCheck:
    reject_reasons: List[str] = []
    review_reasons: List[str] = []

    # These three are window-level, not sentence-level (see module docstring)
    # - real signal, but too coarse to trust for an automatic reject.
    window_level_reasons: List[str] = []
    if utt.no_speech_prob >= no_speech_prob_threshold:
        window_level_reasons.append(f"no_speech_prob_{utt.no_speech_prob:.2f}_gte_{no_speech_prob_threshold}")
    if utt.avg_logprob <= avg_logprob_threshold:
        window_level_reasons.append(f"avg_logprob_{utt.avg_logprob:.2f}_lte_{avg_logprob_threshold}")
    if utt.compression_ratio >= compression_ratio_threshold:
        window_level_reasons.append(f"compression_ratio_{utt.compression_ratio:.2f}_gte_{compression_ratio_threshold}")
    if window_level_reasons:
        review_reasons.extend(f"whisper_window_level_signal:{r}" for r in window_level_reasons)

    # These ARE computed from this exact clip's own transcribed text, so
    # they're trustworthy enough to reject on automatically.
    if _has_repeated_ngram(utt.text):
        reject_reasons.append("repeated_ngram_pattern_detected")
    if not utt.text.strip():
        reject_reasons.append("empty_transcript_text")

    return HallucinationCheck(reject_reasons=reject_reasons, review_reasons=review_reasons)
