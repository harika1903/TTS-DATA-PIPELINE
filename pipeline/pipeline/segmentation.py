"""Context-aware segmentation (protocols 11, 12, 16).

Turns Whisper's segment/word timestamps into TTS training-clip candidates:

  * Never cuts mid-word - every clip boundary snaps to a word start/end
    time (plus a small silence pad), never an arbitrary timestamp.
  * Merges adjacent Whisper segments (small gap between them) up into the
    3-12s preferred range so a "clip" is a full meaningful utterance rather
    than one Whisper micro-segment.
  * Any span that would exceed the 20s hard cap is split, but ONLY at an
    internal pause between words - never mid-sentence with no pause. If no
    safe split point exists, the span is left intact and flagged in the run
    notes for human review rather than force-cut.
  * Long internal pauses (> max_internal_pause_sec) are capped down to a
    short natural-sounding pause rather than deleted outright, by splicing
    around the middle of the gap - the cut always lands inside silence.
  * Spans shorter than the minimum are dropped as candidates (not an error -
    they're just not useful enough to keep, and are more sensitive to
    timestamp noise than longer utterances).
"""
from __future__ import annotations

import dataclasses
from typing import List, Tuple

from .transcribe import Segment, TranscriptionResult, Word


@dataclasses.dataclass
class Utterance:
    words: List[Word]
    text: str
    orig_start: float
    orig_end: float
    kept_spans: List[Tuple[float, float]]  # spans of ORIGINAL audio to splice together
    total_kept_duration: float
    internal_pauses_trimmed: int
    source_segment_ids: List[int]
    avg_logprob: float
    no_speech_prob: float
    compression_ratio: float


def _merge_segments(
    segments: List[Segment], max_gap_sec: float, preferred_max: float
) -> List[List[Segment]]:
    """Greedily group adjacent Whisper segments to approach (but not exceed)
    the preferred duration. A segment already longer than `preferred_max` on
    its own starts and ends its own group; the hard-cap splitter handles it
    afterward.
    """
    if not segments:
        return []
    groups: List[List[Segment]] = [[segments[0]]]
    for seg in segments[1:]:
        cur_group = groups[-1]
        last = cur_group[-1]
        gap = seg.start - last.end
        group_start = cur_group[0].start
        would_be_duration = seg.end - group_start
        if gap <= max_gap_sec and would_be_duration <= preferred_max:
            cur_group.append(seg)
        else:
            groups.append([seg])
    return groups


def _flatten_words(segs: List[Segment]) -> List[Word]:
    words: List[Word] = []
    for s in segs:
        words.extend(s.words)
    return words


def _split_at_hard_cap(
    words: List[Word], hard_cap_sec: float, min_clip_sec: float
) -> List[List[Word]]:
    """Recursively split a word span so every piece is <= hard_cap_sec,
    always splitting at the largest internal gap that keeps both halves
    usable. If no internal gap allows a legal split, the whole span is
    returned unsplit (caller flags this for review).
    """
    if not words:
        return []
    duration = words[-1].end - words[0].start
    if duration <= hard_cap_sec:
        return [words]

    gaps = [(i, words[i + 1].start - words[i].end) for i in range(len(words) - 1)]
    best_idx = None
    best_gap = -1.0
    for i, gap in gaps:
        first_half_dur = words[i].end - words[0].start
        second_half_dur = words[-1].end - words[i + 1].start
        if first_half_dur <= hard_cap_sec and first_half_dur >= min_clip_sec and second_half_dur >= min_clip_sec:
            if gap > best_gap:
                best_gap = gap
                best_idx = i

    if best_idx is None:
        return [words]  # no safe split point - leave intact, flagged upstream

    left = words[: best_idx + 1]
    right = words[best_idx + 1:]
    return _split_at_hard_cap(left, hard_cap_sec, min_clip_sec) + \
        _split_at_hard_cap(right, hard_cap_sec, min_clip_sec)


def _trim_pauses_to_spans(
    words: List[Word],
    max_internal_pause_sec: float,
    trimmed_pause_target_sec: float,
    word_pad_sec: float = 0.05,
) -> Tuple[List[Tuple[float, float]], int]:
    """Build the list of ORIGINAL-audio spans to keep. Any internal gap
    longer than max_internal_pause_sec is capped (not deleted) to
    trimmed_pause_target_sec, split evenly on both sides so the cut always
    lands inside silence, never inside a word.
    """
    if not words:
        return [], 0

    spans: List[Tuple[float, float]] = []
    cursor_start = max(0.0, words[0].start - word_pad_sec)
    trimmed_count = 0

    for i in range(len(words) - 1):
        gap = words[i + 1].start - words[i].end
        if gap > max_internal_pause_sec:
            half_keep = trimmed_pause_target_sec / 2.0
            cut_end = words[i].end + half_keep
            cut_resume = words[i + 1].start - half_keep
            if cut_resume <= cut_end:
                continue  # gap too tight to safely trim; leave it as natural pause
            spans.append((cursor_start, cut_end))
            cursor_start = cut_resume
            trimmed_count += 1

    spans.append((cursor_start, words[-1].end + word_pad_sec))
    return spans, trimmed_count


def build_utterances(
    transcription: TranscriptionResult,
    min_clip_sec: float,
    preferred_max_sec: float,
    hard_cap_sec: float,
    max_silence_between_merge_sec: float,
    max_internal_pause_sec: float,
    trimmed_pause_target_sec: float = 0.4,
) -> Tuple[List[Utterance], List[str]]:
    """Main segmentation entry point.

    Returns (utterances, notes); `notes` records file-level segmentation
    observations (e.g. spans with no safe split point) for the run report.
    """
    notes: List[str] = []
    segments = [s for s in transcription.segments if s.words]
    groups = _merge_segments(segments, max_silence_between_merge_sec, preferred_max_sec)

    utterances: List[Utterance] = []
    for group in groups:
        words = _flatten_words(group)
        if not words:
            continue
        word_chunks = _split_at_hard_cap(words, hard_cap_sec, min_clip_sec)
        for chunk in word_chunks:
            if not chunk:
                continue
            dur = chunk[-1].end - chunk[0].start
            if dur > hard_cap_sec:
                notes.append(
                    f"segment_around_{chunk[0].start:.1f}s_exceeds_hard_cap_no_safe_split_point"
                )
            if dur < min_clip_sec:
                continue  # not an error - just not kept as a candidate

            spans, trimmed = _trim_pauses_to_spans(
                chunk, max_internal_pause_sec, trimmed_pause_target_sec
            )
            kept_duration = sum(e - s for s, e in spans)
            text = "".join(w.word for w in chunk).strip()

            src_ids = sorted({s.id for s in group})
            weights = [max(0.001, s.end - s.start) for s in group]
            avg_logprob = sum(s.avg_logprob * w for s, w in zip(group, weights)) / sum(weights)
            no_speech_prob = max(s.no_speech_prob for s in group)
            compression_ratio = max(s.compression_ratio for s in group)

            utterances.append(Utterance(
                words=chunk,
                text=text,
                orig_start=chunk[0].start,
                orig_end=chunk[-1].end,
                kept_spans=spans,
                total_kept_duration=kept_duration,
                internal_pauses_trimmed=trimmed,
                source_segment_ids=src_ids,
                avg_logprob=avg_logprob,
                no_speech_prob=no_speech_prob,
                compression_ratio=compression_ratio,
            ))

    return utterances, notes
