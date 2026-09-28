"""Transcription + voice-activity detection via faster-whisper (CTranslate2).

WHY faster-whisper INSTEAD OF openai-whisper / torchaudio:
  * CTranslate2 gives a large (typically 4-8x) CPU speedup over the
    reference openai-whisper implementation, and supports int8 quantization
    on CPU - this directly addresses "Whisper `small` on CPU was far too
    slow on a 90-minute file".
  * It has no torchaudio/torchcodec dependency at all (the earlier
    implementation's "torchcodec missing" failure came from a torchaudio
    version that expected it) - faster-whisper decodes audio itself via
    ffmpeg/PyAV, so that failure mode is structurally avoided here.
  * Its `vad_filter=True` option runs the bundled Silero VAD (ONNX) to skip
    non-speech BEFORE transcription, so VAD (protocol 4) and transcription
    (protocol 9) share a single model pass over the file instead of being
    two independent full-file passes.

This module performs exactly one transcribe() call per source file; its
output (segments + word timestamps + per-segment confidence/hallucination
signals) is cached to disk and reused by every later stage.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import time
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger("tts_pipeline.transcribe")

try:
    from faster_whisper import WhisperModel
except ImportError:  # pragma: no cover - surfaced clearly at call time instead
    WhisperModel = None  # type: ignore


@dataclasses.dataclass
class Word:
    word: str
    start: float
    end: float
    probability: float


@dataclasses.dataclass
class Segment:
    id: int
    start: float
    end: float
    text: str
    avg_logprob: float
    no_speech_prob: float
    compression_ratio: float
    words: List[Word]


@dataclasses.dataclass
class TranscriptionResult:
    language: str
    language_probability: float
    duration_sec: float
    segments: List[Segment]
    model_name: str
    device: str
    compute_type: str
    # (start, end) sec per Whisper segment - effectively the VAD-filtered
    # speech regions, since vad_filter=True already dropped non-speech.
    vad_speech_regions: List[Tuple[float, float]]
    elapsed_sec: float


_MODEL_CACHE: dict = {}


def get_model(model_name: str, device: str, compute_type: str) -> "WhisperModel":
    if WhisperModel is None:
        raise RuntimeError(
            "faster-whisper is not installed. Install it with:\n"
            "    pip install faster-whisper\n"
            "(This pipeline intentionally does not depend on openai-whisper "
            "or torchaudio for transcription.)"
        )
    key = (model_name, device, compute_type)
    if key not in _MODEL_CACHE:
        logger.info("Loading faster-whisper model=%s device=%s compute_type=%s ...",
                    model_name, device, compute_type)
        t0 = time.time()
        _MODEL_CACHE[key] = WhisperModel(model_name, device=device, compute_type=compute_type)
        logger.info("Model loaded in %.1fs", time.time() - t0)
    return _MODEL_CACHE[key]


def resolve_device_and_compute_type(
    requested_device: str, cpu_compute: str, gpu_compute: str
) -> Tuple[str, str]:
    if requested_device == "cuda":
        return "cuda", gpu_compute
    if requested_device == "cpu":
        return "cpu", cpu_compute
    # auto
    try:
        import torch  # optional; only used to probe CUDA availability
        if torch.cuda.is_available():
            return "cuda", gpu_compute
    except ImportError:
        pass
    return "cpu", cpu_compute


def transcribe_file(
    analysis_wav: Path,
    model_name: str,
    device: str,
    compute_type: str,
    language: Optional[str],
    vad_min_silence_ms: int = 200,
    vad_speech_pad_ms: int = 100,
    beam_size: int = 5,
    progress_log_every_sec: float = 30.0,
) -> TranscriptionResult:
    """Run the single transcription + VAD pass with word-level timestamps.

    faster-whisper's `transcribe()` returns a lazy generator; segments are
    only produced as the audio is actually decoded and run through the
    model. We log progress as segments arrive (by audio position, not just
    segment count) specifically so a long file does not look "stuck at 0%"
    the way the earlier implementation could.
    """
    model = get_model(model_name, device, compute_type)

    t0 = time.time()
    segments_iter, info = model.transcribe(
        str(analysis_wav),
        language=language,
        beam_size=beam_size,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=vad_min_silence_ms,
            speech_pad_ms=vad_speech_pad_ms,
        ),
        condition_on_previous_text=False,  # avoid run-on hallucination propagation
    )

    segments: List[Segment] = []
    speech_regions: List[Tuple[float, float]] = []
    last_log = t0
    for i, seg in enumerate(segments_iter):
        words = [
            Word(word=w.word, start=w.start, end=w.end, probability=w.probability)
            for w in (seg.words or [])
        ]
        segments.append(Segment(
            id=i,
            start=seg.start,
            end=seg.end,
            text=seg.text.strip(),
            avg_logprob=seg.avg_logprob,
            no_speech_prob=seg.no_speech_prob,
            compression_ratio=seg.compression_ratio,
            words=words,
        ))
        speech_regions.append((seg.start, seg.end))

        now = time.time()
        if now - last_log >= progress_log_every_sec:
            logger.info(
                "  transcribing... reached audio position %.1fs (%d segments so far, %.1fs elapsed)",
                seg.end, len(segments), now - t0,
            )
            last_log = now

    elapsed = time.time() - t0
    logger.info("Transcription finished: %d segments, %.1fs elapsed, language=%s (p=%.2f)",
                len(segments), elapsed, info.language, info.language_probability)

    return TranscriptionResult(
        language=info.language,
        language_probability=info.language_probability,
        duration_sec=info.duration,
        segments=segments,
        model_name=model_name,
        device=device,
        compute_type=compute_type,
        vad_speech_regions=speech_regions,
        elapsed_sec=elapsed,
    )


def save_transcription(result: TranscriptionResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dataclasses.asdict(result)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def load_transcription(path: Path) -> Optional[TranscriptionResult]:
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text())
        segments = [
            Segment(
                id=s["id"], start=s["start"], end=s["end"], text=s["text"],
                avg_logprob=s["avg_logprob"], no_speech_prob=s["no_speech_prob"],
                compression_ratio=s["compression_ratio"],
                words=[Word(**w) for w in s["words"]],
            )
            for s in raw["segments"]
        ]
        return TranscriptionResult(
            language=raw["language"],
            language_probability=raw["language_probability"],
            duration_sec=raw["duration_sec"],
            segments=segments,
            model_name=raw["model_name"],
            device=raw.get("device", "unknown"),
            compute_type=raw.get("compute_type", "unknown"),
            vad_speech_regions=[tuple(r) for r in raw["vad_speech_regions"]],
            elapsed_sec=raw.get("elapsed_sec", 0.0),
        )
    except Exception as e:  # noqa: BLE001 - corrupt cache should not crash the run
        logger.warning("Failed to load cached transcription %s (%s); will re-transcribe.", path, e)
        return None
