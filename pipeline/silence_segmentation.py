"""Silence-based audio segmentation (approach B -- no timestamps needed).

Because the custom STT server returns text with NO timestamps, we cannot cut
clips at word boundaries the way the Whisper pipeline does. Instead we cut at
SILENCE: find the quiet gaps in the audio (natural pauses between phrases) and
cut there, then transcribe each already-short clip.

HOW IT WORKS:
  1. Compute a short-time energy envelope over the audio.
  2. Mark frames below a silence threshold as "silence".
  3. Find silence runs long enough to be real pauses (min_silence_sec).
  4. Cut at the MIDDLE of each qualifying pause -- so the cut lands squarely
     inside quiet, never clipping the start/end of a word.
  5. Merge resulting pieces toward the 3-12s target; hard-split only if a
     piece exceeds the 20s cap AND contains an internal pause to split at.

HONEST LIMITATIONS (the tradeoff the user accepted):
  * Context depends on the speaker pausing at meaningful boundaries. Natural
    speech usually does; run-on speech (few pauses) yields long clips that may
    exceed the cap with no safe split point -> those get flagged/rejected.
  * A speaker who pauses mid-thought (hesitation) can produce a clip that cuts
    an incomplete phrase. There is no way to detect this without knowing the
    words -- which we don't, since there are no timestamps.
  * This is inherently less precise than Whisper's word-level cutting. It is
    the best that can be done when the STT model provides no timing.
"""
from __future__ import annotations

import dataclasses
from typing import List, Tuple

import numpy as np


@dataclasses.dataclass
class SilenceSpan:
    start_sec: float
    end_sec: float
    duration: float


def _frame_energy_db(samples: np.ndarray, sr: int, frame_ms: float, hop_ms: float):
    """Return (times, energy_db) for short frames across the whole signal."""
    frame_len = max(1, int(sr * frame_ms / 1000.0))
    hop_len = max(1, int(sr * hop_ms / 1000.0))
    n = samples.shape[0]
    times = []
    energies = []
    for start in range(0, max(1, n - frame_len + 1), hop_len):
        frame = samples[start:start + frame_len]
        rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2) + 1e-12))
        db = 20.0 * np.log10(max(rms, 1e-9))
        times.append(start / sr)
        energies.append(db)
    return np.array(times), np.array(energies)


def find_cut_points(
    samples: np.ndarray,
    sr: int,
    silence_db_threshold: float = -38.0,
    min_silence_sec: float = 0.35,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
) -> List[float]:
    """Return a sorted list of times (seconds) at which to cut -- the MIDDLE
    of each qualifying silence run. `samples` should be mono float32.

    silence_db_threshold: frames quieter than this are "silence". -38 dBFS is
    a reasonable default for speech recorded at normal levels; raise toward
    -30 for noisier audio, lower toward -45 for very clean/quiet audio.
    min_silence_sec: only silence runs at least this long count as a real
    pause (avoids cutting at tiny gaps between syllables).
    """
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    times, energy_db = _frame_energy_db(samples, sr, frame_ms, hop_ms)
    if len(times) == 0:
        return []

    is_silence = energy_db < silence_db_threshold

    cut_points: List[float] = []
    run_start_idx = None
    for i, silent in enumerate(is_silence):
        if silent and run_start_idx is None:
            run_start_idx = i
        elif not silent and run_start_idx is not None:
            run_start_t = times[run_start_idx]
            run_end_t = times[i]
            if run_end_t - run_start_t >= min_silence_sec:
                cut_points.append((run_start_t + run_end_t) / 2.0)
            run_start_idx = None
    # trailing silence run
    if run_start_idx is not None:
        run_start_t = times[run_start_idx]
        run_end_t = times[-1]
        if run_end_t - run_start_t >= min_silence_sec:
            cut_points.append((run_start_t + run_end_t) / 2.0)

    return cut_points


@dataclasses.dataclass
class SilenceClip:
    start_sec: float
    end_sec: float
    duration: float


def build_clips_from_cuts(
    total_duration_sec: float,
    cut_points: List[float],
    min_clip_sec: float,
    preferred_max_sec: float,
    hard_cap_sec: float,
    pad_sec: float = 0.10,
    end_pad_sec: float = 0.40,
) -> Tuple[List[SilenceClip], List[str]]:
    """Turn cut points into clip spans, targeting min..preferred and never
    exceeding the hard cap where avoidable.

    Strategy:
      * Start with the raw pieces between consecutive cut points.
      * Greedily merge adjacent pieces while the merged length stays <=
        preferred_max_sec (so short fragments become full utterances).
      * Drop pieces shorter than min_clip_sec (too short to be useful).
      * Any piece still longer than hard_cap_sec has no internal pause to
        split at (all its internal cuts were already used) -> flag it.

    Padding: `pad_sec` is added at the START, and `end_pad_sec` (larger) at
    the END. The larger trailing pad captures words that trail off softly --
    a word ending quietly can extend a little past where the silence detector
    marked the cut, so without extra trailing audio the last word gets its
    tail clipped (which is why the last word sometimes went missing). Cut
    points are the MIDDLE of a silence, so there is quiet room after the last
    word to extend into without grabbing the next word.

    Returns (clips, notes).
    """
    notes: List[str] = []
    boundaries = [0.0] + sorted(cut_points) + [total_duration_sec]
    # raw pieces
    raw: List[SilenceClip] = []
    for i in range(len(boundaries) - 1):
        s = boundaries[i]
        e = boundaries[i + 1]
        if e - s > 0.05:
            raw.append(SilenceClip(s, e, e - s))

    # greedy merge toward preferred_max
    merged: List[SilenceClip] = []
    cur_start = None
    cur_end = None
    for piece in raw:
        if cur_start is None:
            cur_start, cur_end = piece.start_sec, piece.end_sec
            continue
        if (piece.end_sec - cur_start) <= preferred_max_sec:
            cur_end = piece.end_sec  # extend current clip
        else:
            merged.append(SilenceClip(cur_start, cur_end, cur_end - cur_start))
            cur_start, cur_end = piece.start_sec, piece.end_sec
    if cur_start is not None:
        merged.append(SilenceClip(cur_start, cur_end, cur_end - cur_start))

    # finalize: pad (small at start, larger at end to catch trailing words),
    # drop too-short, flag too-long
    clips: List[SilenceClip] = []
    for c in merged:
        start = max(0.0, c.start_sec - pad_sec)
        end = min(total_duration_sec, c.end_sec + end_pad_sec)
        dur = end - start
        if dur < min_clip_sec:
            continue  # too short to be useful -- not an error
        if dur > hard_cap_sec + 0.05:
            notes.append(f"silence_clip_{start:.1f}s_to_{end:.1f}s_is_{dur:.1f}s_no_internal_pause_to_split")
            # keep it anyway; the per-clip duration gate downstream will reject it
        clips.append(SilenceClip(start, end, dur))

    return clips, notes
