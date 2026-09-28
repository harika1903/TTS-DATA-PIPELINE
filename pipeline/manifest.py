"""Output manifest and dataset-level reports (protocols 19, 20, 21, 22).

Also provides light-touch implementations of protocol 20 (phoneme coverage)
and 21 (split / leakage control) at the DATASET level, run once after all
source files are processed, since both are cross-file concerns rather than
per-clip ones.
"""
from __future__ import annotations

import dataclasses
import json
import random
import uuid
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

PIPELINE_VERSION = "1.0.0"


@dataclasses.dataclass
class ClipRecord:
    utterance_id: str
    status: str  # "accepted" | "quarantined" | "rejected"
    source_id: str
    source_filename: str
    start_sec: float
    end_sec: float
    duration_sec: float
    text: str
    audio_path: Optional[str]
    detected_language: str
    language_probability: float
    avg_logprob: float
    no_speech_prob: float
    compression_ratio: float
    mean_word_confidence: float
    chars_per_sec: float
    est_snr_db: float
    background_verdict: str
    clipped_sample_ratio: float
    speaker_verification_method: str
    speaker_verified: bool
    loudness_applied: bool
    input_lufs: Optional[float]
    output_lufs: Optional[float]
    internal_pauses_trimmed: int
    needs_human_review: bool
    needs_number_review: bool
    rejection_reasons: List[str]
    review_reasons: List[str]
    dnsmos_sig: Optional[float] = None
    dnsmos_bak: Optional[float] = None
    dnsmos_ovrl: Optional[float] = None
    pipeline_version: str = PIPELINE_VERSION


def new_utterance_id(source_id: str) -> str:
    return f"{source_id}__{uuid.uuid4().hex[:12]}"


def append_record(record: ClipRecord, manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(dataclasses.asdict(record), ensure_ascii=False) + "\n")


def load_records(manifest_path: Path) -> List[dict]:
    if not manifest_path.exists():
        return []
    out = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def rewrite_records(records: List[dict], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _count_reasons(records: List[dict], field_name: str = "rejection_reasons") -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for r in records:
        for reason in r.get(field_name, []):
            counts[reason] = counts.get(reason, 0) + 1
    return counts


def write_dataset_summary(
    reports_dir: Path, accepted: List[dict], quarantined: List[dict], rejected: List[dict]
) -> Path:
    total_accepted_dur = sum(r["duration_sec"] for r in accepted)
    summary = {
        "pipeline_version": PIPELINE_VERSION,
        "counts": {
            "accepted": len(accepted),
            "quarantined": len(quarantined),
            "rejected": len(rejected),
            "total_candidates": len(accepted) + len(quarantined) + len(rejected),
        },
        "accepted_total_duration_sec": total_accepted_dur,
        "accepted_total_duration_hours": total_accepted_dur / 3600.0,
        "rejection_reason_counts": _count_reasons(rejected, "rejection_reasons"),
        "quarantine_reason_counts": _count_reasons(quarantined, "review_reasons"),
        "unique_source_files": len({r["source_id"] for r in accepted + quarantined + rejected}),
    }
    out_path = reports_dir / "dataset_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return out_path


def phoneme_coverage_report(reports_dir: Path, accepted: List[dict]) -> Path:
    """RELIABILITY NOTE: true phoneme coverage requires a grapheme-to-
    phoneme (G2P) model for the target language, which this pipeline does
    not bundle (accuracy and availability vary a lot by language, and none
    is safe to assume installed). What's computed here instead is a
    CHARACTER-level coverage proxy, clearly labeled as such, so obvious gaps
    (e.g. a letter/digraph that never appears in the accepted set) are at
    least visible. For real phoneme-level coverage, run a G2P pass (e.g.
    `g2p_en` for English) over `accepted[*]["text"]` and swap this function's
    counters for phoneme counters.
    """
    letters: Counter = Counter()
    bigrams: Counter = Counter()
    for r in accepted:
        text = r["text"].lower()
        for ch in text:
            if ch.isalpha():
                letters[ch] += 1
        for i in range(len(text) - 1):
            bg = text[i:i + 2]
            if bg[0].isalpha() and (bg[1].isalpha() or bg[1] == " "):
                bigrams[bg] += 1

    report = {
        "note": (
            "Character-level coverage proxy only, NOT true phoneme coverage. "
            "See phoneme_coverage_report() docstring for how to upgrade to a real G2P-based report."
        ),
        "unique_letters_seen": len(letters),
        "letter_counts": dict(letters.most_common()),
        "top_50_bigrams": dict(bigrams.most_common(50)),
    }
    out_path = reports_dir / "phoneme_coverage_proxy.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return out_path


def split_dataset(
    accepted: List[dict],
    reports_dir: Path,
    train_ratio: float = 0.9,
    val_ratio: float = 0.05,
    seed: int = 42,
) -> Path:
    """Splits at the SOURCE-FILE level (not the individual-clip level) so
    every clip from a given source recording (and therefore, presumably, a
    given speaker/session) lands in exactly one split - preventing
    speaker/recording leakage between train/val/test (protocol 21).
    """
    by_source: Dict[str, List[dict]] = {}
    for r in accepted:
        by_source.setdefault(r["source_id"], []).append(r)

    source_ids = list(by_source.keys())
    rng = random.Random(seed)
    rng.shuffle(source_ids)

    n = len(source_ids)
    if n <= 2:
        # Too few source files to meaningfully split; everything goes to train
        # and this is called out explicitly rather than producing an empty
        # val/test split silently.
        split_map = {sid: "train" for sid in source_ids}
    else:
        n_train = max(1, int(round(n * train_ratio)))
        n_val = max(0, int(round(n * val_ratio)))
        n_train = min(n_train, n - 1)  # leave at least one source for val/test
        train_ids = source_ids[:n_train]
        val_ids = source_ids[n_train:n_train + n_val]
        test_ids = source_ids[n_train + n_val:]
        split_map = {}
        split_map.update({sid: "train" for sid in train_ids})
        split_map.update({sid: "val" for sid in val_ids})
        split_map.update({sid: "test" for sid in test_ids})

    counts = {"train": 0, "val": 0, "test": 0}
    clip_counts = {"train": 0, "val": 0, "test": 0}
    for sid, split in split_map.items():
        counts[split] += 1
        clip_counts[split] += len(by_source[sid])

    out = {
        "seed": seed,
        "split_by_source_id": split_map,
        "note": "Split at source-file granularity to prevent speaker/recording leakage between train/val/test.",
        "source_counts": counts,
        "clip_counts": clip_counts,
        "too_few_sources_for_split": n <= 2,
    }
    out_path = reports_dir / "dataset_split.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    return out_path
