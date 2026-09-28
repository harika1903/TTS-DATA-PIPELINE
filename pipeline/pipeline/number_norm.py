"""Number normalization via NVIDIA NeMo (protocol: numbers as actually spoken).

NeMo's text-normalization (Apache 2.0) can render numbers/dates the way they
were spoken. This module wraps it defensively and, crucially, is
LANGUAGE-AWARE: NeMo only supports some languages, so `normalize` takes the
per-file language and returns `applied=False` (leaving text untouched) for
any language NeMo doesn't cover, instead of silently mangling it.

Per the pipeline's philosophy: when in doubt, DON'T rewrite. Auto-"fixing" a
number into a written form that no longer matches what was said would break
alignment. So for unsupported languages, digit-containing clips are left
as-is and flagged for human review (the base pipeline already does this).

NOT run in development. First real run is the true test. Also note: full
NeMo is a heavy install; if you only need text normalization, the lighter
`nemo_text_processing` package is enough.
"""
from __future__ import annotations

import dataclasses
import logging
from typing import Optional

logger = logging.getLogger("tts_pipeline.nemo_norm")


@dataclasses.dataclass
class NormalizationResult:
    applied: bool
    text: str
    note: str = ""


_NORMALIZERS: dict = {}


def _get_normalizer(nemo_lang: str):
    if nemo_lang not in _NORMALIZERS:
        from nemo_text_processing.text_normalization.normalize import Normalizer
        _NORMALIZERS[nemo_lang] = Normalizer(input_case="cased", lang=nemo_lang)
    return _NORMALIZERS[nemo_lang]


def normalize(text: str, nemo_lang: Optional[str]) -> NormalizationResult:
    """Normalize numbers in `text` for the given NeMo language code. If
    nemo_lang is None (language unsupported) or NeMo isn't installed, returns
    the text unchanged with applied=False -- never raises.
    """
    if nemo_lang is None:
        return NormalizationResult(applied=False, text=text, note="language not supported by NeMo; text left unchanged")

    try:
        normalizer = _get_normalizer(nemo_lang)
    except ImportError as e:
        return NormalizationResult(applied=False, text=text, note=f"nemo_text_processing not installed ({e}); text unchanged")
    except Exception as e:  # noqa: BLE001
        return NormalizationResult(applied=False, text=text, note=f"NeMo normalizer init failed ({e}); text unchanged")

    try:
        normalized = normalizer.normalize(text, verbose=False)
        return NormalizationResult(applied=True, text=normalized)
    except Exception as e:  # noqa: BLE001
        logger.warning("NeMo normalization failed on a clip (%s); text left unchanged.", e)
        return NormalizationResult(applied=False, text=text, note=f"NeMo runtime error ({e}); text unchanged")
