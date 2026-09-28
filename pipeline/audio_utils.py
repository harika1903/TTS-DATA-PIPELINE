"""Audio I/O, integrity checking, format conversion, and streaming acoustic
metrics.

Every function here is written to avoid loading an entire (potentially
1GB+, 90-minute) audio file into memory at once, and to avoid decoding the
same file more than once:

  * `ffprobe_metadata` / `verify_integrity` never decode audio at all - they
    only parse the container/stream headers.
  * `make_analysis_copy` runs ffmpeg exactly once per source file to produce
    a throwaway mono/16kHz copy, and is skipped entirely if that copy
    already exists on disk from a previous run.
  * `compute_streaming_metrics` performs exactly ONE streaming pass over the
    (small, 16kHz mono) analysis copy, in fixed-size blocks, and caches a
    per-block RMS trace that `background.py` reuses later instead of
    re-reading the file for noise-floor estimation.
  * `read_segment` / `cut_clip_from_spans` seek directly to the requested
    time range in the ORIGINAL (full quality, full sample rate) file rather
    than decoding it from the start, so cutting clip #500 out of a 90-minute
    recording costs the same as cutting clip #1.
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf

logger = logging.getLogger("tts_pipeline.audio")


class AudioIntegrityError(Exception):
    """Raised when a source file (or an analysis artifact derived from it)
    fails an integrity check. Files raising this should be quarantined at
    the whole-file level, not passed further into the pipeline.
    """


@dataclass
class SourceMetadata:
    path: str
    codec: str
    channels: int
    sample_rate: int
    bits_per_sample: Optional[int]
    duration_sec: float
    size_bytes: int
    has_audio_stream: bool


def _run(cmd: Sequence[str]) -> subprocess.CompletedProcess:
    return subprocess.run(list(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)


def ffprobe_metadata(path: Path) -> SourceMetadata:
    """Run ffprobe once to fetch stream metadata (protocol 1: source audio
    integrity, part 1). Raises AudioIntegrityError if the file is not
    parseable or has no audio stream. This is a header-only inspection - it
    does not decode audio samples, so it is cheap even for a 1GB file.
    """
    if shutil.which("ffprobe") is None:
        raise RuntimeError("ffprobe not found on PATH. Install ffmpeg (`apt install ffmpeg`).")

    cmd = [
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ]
    proc = _run(cmd)
    if proc.returncode != 0:
        raise AudioIntegrityError(
            f"ffprobe failed for {path}: {proc.stderr.decode(errors='ignore')[:500]}"
        )
    try:
        info = json.loads(proc.stdout.decode(errors="ignore"))
    except json.JSONDecodeError as e:
        raise AudioIntegrityError(f"ffprobe returned invalid JSON for {path}: {e}")

    audio_streams = [s for s in info.get("streams", []) if s.get("codec_type") == "audio"]
    if not audio_streams:
        raise AudioIntegrityError(
            f"No audio stream found in {path} (ffprobe reported 0 audio streams in the container)."
        )

    astream = audio_streams[0]
    fmt = info.get("format", {})

    duration_raw = astream.get("duration") or fmt.get("duration")
    try:
        duration = float(duration_raw) if duration_raw is not None else 0.0
    except (TypeError, ValueError):
        duration = 0.0

    bits_raw = astream.get("bits_per_sample") or astream.get("bits_per_raw_sample")
    try:
        bits = int(bits_raw) if bits_raw else None
    except (TypeError, ValueError):
        bits = None

    return SourceMetadata(
        path=str(path),
        codec=astream.get("codec_name", "unknown"),
        channels=int(astream.get("channels", 0) or 0),
        sample_rate=int(astream.get("sample_rate", 0) or 0),
        bits_per_sample=bits,
        duration_sec=duration,
        size_bytes=int(fmt.get("size", path.stat().st_size)),
        has_audio_stream=True,
    )


def verify_integrity(path: Path, meta: SourceMetadata) -> None:
    """Sanity-check ffprobe's metadata (protocol 1, part 2). Raises
    AudioIntegrityError on any hard failure.

    NOTE ON RELIABILITY: this does not perform a full sample-by-sample
    decode of the source (that would mean two full reads of a 1GB file just
    to validate it, before doing any real work). A file that passes this
    check but is corrupted partway through will still be caught -
    `compute_streaming_metrics` decodes the analysis copy end-to-end and
    will raise if decoding breaks mid-file, which is the one full-file
    decode this pipeline performs.
    """
    if meta.sample_rate <= 0:
        raise AudioIntegrityError(f"Invalid sample rate reported for {path}: {meta.sample_rate}")
    if meta.channels <= 0:
        raise AudioIntegrityError(f"Invalid channel count reported for {path}: {meta.channels}")
    if meta.duration_sec <= 0.0:
        raise AudioIntegrityError(f"Invalid/zero duration reported for {path}: {meta.duration_sec}")
    if meta.size_bytes < 1024:
        raise AudioIntegrityError(f"File suspiciously small ({meta.size_bytes} bytes): {path}")


def make_analysis_copy(src: Path, dst: Path, sample_rate: int = 16000) -> Path:
    """Create a mono, `sample_rate`-Hz PCM16 WAV analysis copy of `src` via
    ffmpeg (protocol 3). Skipped if `dst` already exists (resumability).

    IMPORTANT: the ORIGINAL file is never modified or re-encoded for the
    final dataset. This downsampled copy exists purely to make VAD,
    transcription, and metric computation cheaper; final clips are always
    cut from the ORIGINAL audio (see `cut_clip_from_spans`), so no quality
    is lost in the delivered dataset.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    tmp = dst.with_suffix(".tmp.wav")
    cmd = [
        "ffmpeg", "-y", "-nostdin", "-i", str(src),
        "-ac", "1", "-ar", str(sample_rate),
        "-c:a", "pcm_s16le",
        str(tmp),
    ]
    proc = _run(cmd)
    if proc.returncode != 0 or not tmp.exists():
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise AudioIntegrityError(
            f"ffmpeg failed to create analysis copy of {src}: "
            f"{proc.stderr.decode(errors='ignore')[-1000:]}"
        )
    tmp.replace(dst)
    return dst


@dataclass
class StreamingMetrics:
    duration_sec: float = 0.0
    sample_rate: int = 0
    rms_dbfs: float = -120.0
    peak_dbfs: float = -120.0
    clipped_sample_ratio: float = 0.0
    silence_ratio: float = 0.0
    block_rms_dbfs: List[float] = field(default_factory=list)
    block_hop_sec: float = 0.0
    n_samples: int = 0


def compute_streaming_metrics(
    wav_path: Path,
    block_seconds: float = 1.0,
    silence_dbfs_threshold: float = -45.0,
    clip_amplitude_threshold: float = 0.998,
) -> StreamingMetrics:
    """Single streaming pass over the (mono, 16kHz) analysis copy, computing
    level, clipping, and silence statistics without holding the whole array
    in memory (protocol 5, whole-file part). This is the one full-file read
    the pipeline performs against the analysis copy; the per-block RMS trace
    it returns is reused by `background.py` for noise-floor estimation
    instead of a second pass.
    """
    info = sf.info(str(wav_path))
    sr = info.samplerate
    block_frames = max(1, int(block_seconds * sr))

    sum_sq = 0.0
    n_total = 0
    peak = 0.0
    clipped = 0
    silent_frames = 0
    block_rms: List[float] = []

    with sf.SoundFile(str(wav_path)) as f:
        for block in f.blocks(blocksize=block_frames, dtype="float32", always_2d=False):
            block = np.asarray(block)
            if block.ndim > 1:
                block = block.mean(axis=1)
            n = block.shape[0]
            if n == 0:
                continue
            n_total += n
            sum_sq += float(np.sum(block.astype(np.float64) ** 2))
            block_peak = float(np.max(np.abs(block)))
            peak = max(peak, block_peak)
            clipped += int(np.sum(np.abs(block) >= clip_amplitude_threshold))

            block_rms_val = float(np.sqrt(np.mean(block.astype(np.float64) ** 2) + 1e-12))
            block_dbfs = 20.0 * np.log10(max(block_rms_val, 1e-9))
            block_rms.append(block_dbfs)
            if block_dbfs < silence_dbfs_threshold:
                silent_frames += n

    if n_total == 0:
        raise AudioIntegrityError(f"Analysis copy contains zero samples: {wav_path}")

    overall_rms = float(np.sqrt(sum_sq / n_total))
    rms_dbfs = 20.0 * np.log10(max(overall_rms, 1e-9))
    peak_dbfs = 20.0 * np.log10(max(peak, 1e-9))

    return StreamingMetrics(
        duration_sec=n_total / sr,
        sample_rate=sr,
        rms_dbfs=rms_dbfs,
        peak_dbfs=peak_dbfs,
        clipped_sample_ratio=clipped / n_total,
        silence_ratio=silent_frames / n_total,
        block_rms_dbfs=block_rms,
        block_hop_sec=block_seconds,
        n_samples=n_total,
    )


def save_streaming_metrics(m: StreamingMetrics, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dataclasses_asdict_safe(m), indent=2))


def dataclasses_asdict_safe(obj) -> dict:
    import dataclasses as _dc
    return _dc.asdict(obj)


def load_streaming_metrics(path: Path) -> Optional[StreamingMetrics]:
    if not path.exists():
        return None
    raw = json.loads(path.read_text())
    return StreamingMetrics(**raw)


def read_segment(path: Path, start_sec: float, end_sec: float) -> Tuple[np.ndarray, int]:
    """Read only the requested time range from disk via seek + partial read
    (never the whole file). Used for per-clip analysis.
    """
    with sf.SoundFile(str(path)) as f:
        sr = f.samplerate
        start_frame = max(0, int(start_sec * sr))
        end_frame = min(f.frames, int(end_sec * sr))
        if end_frame <= start_frame:
            return np.zeros(0, dtype=np.float32), sr
        f.seek(start_frame)
        data = f.read(end_frame - start_frame, dtype="float32", always_2d=False)
        return np.asarray(data), sr


def cut_clip_from_spans(
    original_path: Path,
    spans: Sequence[Tuple[float, float]],
    out_path: Optional[Path] = None,
) -> Tuple[np.ndarray, int]:
    """Splice the given (start, end) spans of the ORIGINAL (full-quality,
    original sample rate/channel count) audio together in memory, and
    optionally write the result to `out_path`.

    Spans are chosen by segmentation.py to fall on word boundaries or inside
    silence, never mid-word, so every splice point here is a cut inside
    silence (or at natural word padding) rather than a hard cut through
    speech. Returns (samples, sample_rate); `samples` may be multi-channel.
    """
    pieces = []
    sr = None
    with sf.SoundFile(str(original_path)) as f:
        sr = f.samplerate
        n_frames_total = f.frames
        for start_sec, end_sec in spans:
            start_frame = max(0, int(start_sec * sr))
            end_frame = min(n_frames_total, int(end_sec * sr))
            if end_frame <= start_frame:
                continue
            f.seek(start_frame)
            data = f.read(end_frame - start_frame, dtype="float32", always_2d=False)
            pieces.append(np.asarray(data))
    if not pieces:
        raise AudioIntegrityError(f"No audio produced when cutting clip from {original_path} with spans {spans}")
    clip = np.concatenate(pieces, axis=0)

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), clip, sr, subtype="PCM_16")

    return clip, sr


def to_mono(samples: np.ndarray) -> np.ndarray:
    if samples.ndim > 1:
        return samples.mean(axis=1)
    return samples


def compute_clip_clipping_ratio(samples: np.ndarray, clip_amplitude_threshold: float = 0.998) -> float:
    """Fraction of samples at or above `clip_amplitude_threshold` (protocol
    5's clipping/distortion check, applied per-clip). Cheap - clips are at
    most 20s, so this never re-scans the source file.
    """
    if samples.size == 0:
        return 0.0
    return float(np.mean(np.abs(samples) >= clip_amplitude_threshold))
