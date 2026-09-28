"""Language handling: map the 3-letter filename prefix to the codes each
tool needs.

Filenames look like `hin_1.wav`, `tam_1.wav`, etc. -- a 3-letter ISO-639-3
style prefix, an underscore, then anything. `language_from_filename` reads
that prefix so every file is processed in its own language automatically,
with no need to pass --language per file.

IMPORTANT PER-LANGUAGE CAVEAT: the audio tools (Whisper, Demucs, DNSMOS,
alignment) are language-independent and work for every code below. The TEXT
tools are not: NeMo number-normalization and hunspell spellcheck only cover
some languages. `check_environment.py` reports exactly which. This map just
records the code translations; it does not promise every tool supports
every language.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Optional

# code_in_filename -> (human_name, whisper_lang, nemo_lang_or_None, hunspell_dict_or_None)
# nemo/hunspell set to None where support is unlikely; verify with check_environment.py
# and fill in real dictionary names for your installed .dic/.aff files.
LANGUAGES = {
    "hin": ("Hindi",     "hi", "hi", "hi_IN"),
    "tam": ("Tamil",     "ta", None, "ta_IN"),
    "tel": ("Telugu",    "te", None, "te_IN"),
    "ben": ("Bengali",   "bn", None, "bn_IN"),
    "mar": ("Marathi",   "mr", None, "mr_IN"),
    "guj": ("Gujarati",  "gu", None, "gu_IN"),
    "kan": ("Kannada",   "kn", None, "kn_IN"),
    "mal": ("Malayalam", "ml", None, "ml_IN"),
    "pan": ("Punjabi",   "pa", None, "pa_IN"),
    "ori": ("Odia",      "or", None, None),
    "urd": ("Urdu",      "ur", None, "ur_PK"),
    "eng": ("English",   "en", "en", "en_US"),
}

# code_in_filename -> language string the CUSTOM STT server expects.
# The server uses full lowercase English names (confirmed: "hindi", "tamil").
# Adjust any of these if your server expects different strings.
CUSTOM_STT_LANG = {
    "hin": "hindi",
    "tam": "tamil",
    "tel": "telugu",
    "ben": "bengali",
    "mar": "marathi",
    "guj": "gujarati",
    "kan": "kannada",
    "mal": "malayalam",
    "pan": "punjabi",
    "ori": "odia",
    "urd": "urdu",
    "eng": "english",
}


@dataclasses.dataclass
class LanguageInfo:
    code: str            # filename prefix, e.g. "hin"
    name: str            # "Hindi"
    whisper: str         # "hi"
    nemo: Optional[str]  # NeMo normalizer lang, or None if unsupported
    hunspell: Optional[str]  # hunspell dict name, or None if unavailable
    custom_stt: Optional[str]  # custom STT server language string, e.g. "hindi"
    recognized: bool     # False if the prefix wasn't in LANGUAGES


def language_from_filename(path: Path) -> LanguageInfo:
    """Extract the language from a filename like `hin_1.wav`. If the prefix
    isn't recognized, returns recognized=False with whisper=None so the
    caller can decide to auto-detect or skip.
    """
    stem = path.stem
    prefix = stem.split("_", 1)[0].lower() if "_" in stem else stem.lower()

    if prefix in LANGUAGES:
        name, whisper, nemo, hunspell = LANGUAGES[prefix]
        return LanguageInfo(code=prefix, name=name, whisper=whisper, nemo=nemo,
                            hunspell=hunspell, custom_stt=CUSTOM_STT_LANG.get(prefix),
                            recognized=True)

    return LanguageInfo(code=prefix, name=f"unknown({prefix})", whisper=None,
                        nemo=None, hunspell=None, custom_stt=None, recognized=False)
