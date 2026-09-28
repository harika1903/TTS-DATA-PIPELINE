"""Digit -> spoken-number-words, transliterated into the target script.

WHY THIS EXISTS
---------------
The custom STT writes numbers as DIGITS ("17"), but a TTS transcript must
carry the number as it was SPOKEN. NeMo (number_norm.py) can't help for Telugu
and the other Indian languages -- it doesn't support them. This module fills
that gap with a small, DETERMINISTIC converter (no ML): a digit is expanded to
English number-words and each word is mapped to its target-script spelling.

IMPORTANT CAVEAT (read this)
----------------------------
A bare digit does NOT record whether the speaker said the number in English
("seventeen") or in the native language ("padihedu"). This module assumes the
ENGLISH reading transliterated into the target script -- e.g. 21 -> "ట్వెంటీ వన్"
-- which matches English-heavy content (tech/gadgets). Where a speaker actually
used native words, this will not match; those are exactly the clips the
audio-informed check (Whisper) should catch. This is a CONVENTION, not a
recovery of ground truth.

The transliteration tables below are seeded with best-effort spellings. A
native speaker should REVIEW/EDIT them once (each is a small, closed set) --
the mechanism is exact; only the spellings are a matter of taste/dialect.
"""
from __future__ import annotations

import dataclasses
import re
from typing import Dict, List, Optional


# --- English number-word building blocks (closed vocabulary) ----------------
_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
         "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
         "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def english_words(n: int) -> List[str]:
    """Cardinal reading of a non-negative int as English word tokens.
    Uses the Indian numbering system (lakh, crore) since these are Indian-
    language transcripts. 21 -> ['twenty','one']; 1700 -> ['seventeen','hundred'].
    """
    if n < 0:
        return ["minus"] + english_words(-n)
    if n < 20:
        return [_ONES[n]]
    if n < 100:
        t, o = divmod(n, 10)
        return [_TENS[t]] + ([_ONES[o]] if o else [])
    if n < 1000:
        h, r = divmod(n, 100)
        return [_ONES[h], "hundred"] + (english_words(r) if r else [])
    if n < 100000:            # thousands
        th, r = divmod(n, 1000)
        return english_words(th) + ["thousand"] + (english_words(r) if r else [])
    if n < 10000000:          # lakh (Indian system)
        l, r = divmod(n, 100000)
        return english_words(l) + ["lakh"] + (english_words(r) if r else [])
    cr, r = divmod(n, 10000000)   # crore
    return english_words(cr) + ["crore"] + (english_words(r) if r else [])


# --- Target-script spellings of those building blocks -----------------------
# EDIT THESE to taste/dialect. Only the ~35 keys below are ever needed.
TELUGU_WORD: Dict[str, str] = {
    "zero": "జీరో", "one": "వన్", "two": "టూ", "three": "త్రీ", "four": "ఫోర్",
    "five": "ఫైవ్", "six": "సిక్స్", "seven": "సెవెన్", "eight": "ఎయిట్", "nine": "నైన్",
    "ten": "టెన్", "eleven": "ఎలెవన్", "twelve": "ట్వెల్వ్", "thirteen": "థర్టీన్",
    "fourteen": "ఫోర్టీన్", "fifteen": "ఫిఫ్టీన్", "sixteen": "సిక్స్టీన్",
    "seventeen": "సెవెంటీన్", "eighteen": "ఎయిటీన్", "nineteen": "నైంటీన్",
    "twenty": "ట్వెంటీ", "thirty": "థర్టీ", "forty": "ఫోర్టీ", "fifty": "ఫిఫ్టీ",
    "sixty": "సిక్స్టీ", "seventy": "సెవెంటీ", "eighty": "ఎయిటీ", "ninety": "నైంటీ",
    "hundred": "హండ్రెడ్", "thousand": "థౌజండ్", "lakh": "లక్ష", "crore": "కోటి",
    "million": "మిలియన్", "billion": "బిలియన్", "minus": "మైనస్",
}

# Native Telugu number words (the OTHER convention: 21 -> ఇరవై ఒకటి). Provided
# so you can switch conventions or fall back per-clip. Not used by default.
TELUGU_NATIVE: Dict[str, str] = {
    "zero": "సున్నా", "one": "ఒకటి", "two": "రెండు", "three": "మూడు", "four": "నాలుగు",
    "five": "ఐదు", "six": "ఆరు", "seven": "ఏడు", "eight": "ఎనిమిది", "nine": "తొమ్మిది",
    "ten": "పది", "eleven": "పదకొండు", "twelve": "పన్నెండు", "thirteen": "పదమూడు",
    "fourteen": "పద్నాలుగు", "fifteen": "పదిహేను", "sixteen": "పదహారు",
    "seventeen": "పదిహేడు", "eighteen": "పద్దెనిమిది", "nineteen": "పంతొమ్మిది",
    "twenty": "ఇరవై", "thirty": "ముప్పై", "forty": "నలభై", "fifty": "యాభై",
    "sixty": "అరవై", "seventy": "డెబ్బై", "eighty": "ఎనభై", "ninety": "తొంభై",
    "hundred": "వందల", "thousand": "వేల", "lakh": "లక్షల", "crore": "కోట్ల",
    "million": "మిలియన్", "billion": "బిలియన్", "minus": "మైనస్",
}

_TABLES: Dict[str, Dict[str, Dict[str, str]]] = {
    "telugu": {"english": TELUGU_WORD, "native": TELUGU_NATIVE},
}

_INT_RE = re.compile(r"\d+")


@dataclasses.dataclass
class TranslitResult:
    text: str
    changed: bool
    numbers_found: int
    flagged: int = 0
    note: str = ""


def _spell_cardinal(n: int, table: Dict[str, str]) -> str:
    """Whole-number reading: 21 -> 'ట్వెంటీ వన్'."""
    return " ".join(table.get(w, w) for w in english_words(n))


def _spell_digits(digits: str, table: Dict[str, str]) -> str:
    """Digit-by-digit reading: '8123' -> 'ఎయిట్ వన్ టూ త్రీ'."""
    return " ".join(table[_ONES[int(d)]] for d in digits)


def transliterate_numbers(text: str, language: str = "telugu",
                          style: str = "english",
                          digitwise_min_len: int = 8) -> TranslitResult:
    """Transliterate every digit run to its English number-words, written in
    the target script (this is transliteration of the spoken number, NOT a
    translation into the language's own number words):

      * a run with fewer than `digitwise_min_len` digits -> read as one English
        number  (121 -> వన్ హండ్రెడ్ ట్వెంటీ వన్ ; 50000 -> ఫిఫ్టీ థౌజండ్)
      * a run with >= digitwise_min_len digits (phone numbers, long IDs) ->
        read digit by digit (9876543210 -> నైన్ ఎయిట్ సెవెన్ ...), because a
        10-digit phone is never spoken as a single cardinal.

    A number glued to letters (t20, r10, 204v) gets a space so the words don't
    fuse to the letter. `style`='english' (21 -> ట్వెంటీ వన్); 'native' exists
    only as a fallback table and is NOT what this is for.
    Unknown language -> text unchanged (never raises)."""
    lang_tables = _TABLES.get(language.lower())
    if not lang_tables:
        return TranslitResult(text=text, changed=False, numbers_found=0,
                              note=f"no number tables for '{language}'; text unchanged")
    table = lang_tables.get(style, lang_tables["english"])

    count = 0
    out = []
    last = 0
    for m in _INT_RE.finditer(text):
        count += 1
        run = m.group()
        words = _spell_digits(run, table) if len(run) >= digitwise_min_len else _spell_cardinal(int(run), table)
        before = text[m.start() - 1] if m.start() > 0 else ""
        after = text[m.end()] if m.end() < len(text) else ""
        # pad if glued to any non-space char (covers Telugu combining marks,
        # which str.isalnum() misses, and latin letters like t20 / r10)
        if before and not before.isspace():
            words = " " + words
        if after and not after.isspace():
            words = words + " "
        out.append(text[last:m.start()])
        out.append(words)
        last = m.end()
    out.append(text[last:])
    new = "".join(out)
    return TranslitResult(text=new, changed=(new != text), numbers_found=count)
