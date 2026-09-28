"""Transcript-level validation (protocols 2, 14, 15).

Language detection (protocol 2) reuses Whisper's own language-ID output
from the single transcription pass - no separate model/pass is run just to
detect language.

Basic well-formedness (protocol 14) is checked directly. Proper-noun and
number normalization (protocol 15) is deliberately NOT auto-rewritten:
silently "normalizing" a spoken number or name (e.g. "twenty three" ->
"23", or expanding an abbreviation) risks producing a written form that no
longer matches what was actually spoken, which would break the audio-text
alignment protocol 15 is supposed to serve. Instead this pipeline preserves
Whisper's own spoken-form transcription verbatim and flags any clip
containing digits for human review, since those are the cases most likely
to need a manual decision about spoken-vs-written form for your TTS
front-end's text normalizer.
"""
from __future__ import annotations

import dataclasses
import re
from typing import List, Optional

_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_ALPHA_RE = re.compile(r"[^\W\d_]", re.UNICODE)
_DIGIT_RE = re.compile(r"\d")


@dataclasses.dataclass
class TranscriptValidation:
    ok: bool
    needs_number_review: bool
    reasons: List[str]


def validate_transcript(
    text: str,
    detected_language: str,
    target_language: Optional[str],
    min_language_probability: float,
    language_probability: float,
) -> TranscriptValidation:
    reasons: List[str] = []
    stripped = text.strip()

    if not stripped:
        reasons.append("empty_transcript")
    if _CONTROL_CHAR_RE.search(stripped):
        reasons.append("contains_control_characters")
    if stripped and not _ALPHA_RE.search(stripped):
        reasons.append("no_alphabetic_content")

    if target_language and detected_language != target_language:
        reasons.append(f"language_mismatch_detected_{detected_language}_target_{target_language}")
    if language_probability < min_language_probability:
        reasons.append(f"low_language_confidence_{language_probability:.2f}")

    needs_number_review = bool(_DIGIT_RE.search(stripped))

    return TranscriptValidation(
        ok=not reasons,
        needs_number_review=needs_number_review,
        reasons=reasons,
    )
