#!/usr/bin/env python3
"""TTS training-data pipeline entry point.

    python3 tts_pipeline.py --input audios --output results --limit 1
    python3 tts_pipeline.py --input audios --output results --force

See README.md for the full architecture write-up: what each of the 22
agreed protocols maps to in this codebase, which ones are heuristic /
best-effort (and why), and how resumability works.
"""
from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import List, Optional

from pipeline.config import PipelineConfig
from pipeline.state import FileState, is_stale, load_state, save_state, source_id_for
from pipeline import audio_utils
from pipeline import transcribe as transcribe_mod
from pipeline import segmentation as seg_mod
from pipeline import background as bg_mod
from pipeline import speaker as speaker_mod
from pipeline import hallucination as halluc_mod
from pipeline import alignment as align_mod
from pipeline import transcript_validate as tv_mod
from pipeline import loudness as loud_mod
from pipeline import dedup as dedup_mod
from pipeline import manifest as manifest_mod

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".wma", ".opus"}

logger = logging.getLogger("tts_pipeline")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a clean TTS training dataset from long recordings.")
    p.add_argument("--input", required=True, help="Input folder of source audio files.")
    p.add_argument("--output", required=True, help="Output folder for the dataset.")
    p.add_argument("--limit", type=int, default=None, help="Only process the first N source files.")
    p.add_argument("--force", action="store_true", help="Reprocess files even if already completed.")
    p.add_argument("--language", default="en", help="Target language code (e.g. 'en'), or 'auto' to auto-detect.")
    p.add_argument("--model", default="small", help="faster-whisper model name/size (e.g. tiny, base, small, medium, large-v3).")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Compute device for Whisper.")
    p.add_argument("--enable-speaker-diarization", action="store_true",
                    help="Enable optional pyannote-based single-speaker verification (requires --hf-token).")
    p.add_argument("--hf-token", default=None, help="Hugging Face token, required if --enable-speaker-diarization is set.")
    p.add_argument("--no-loudness-normalize", action="store_true", help="Disable loudness normalization.")
    p.add_argument("--enable-dnsmos", action="store_true",
                    help="Enable DNSMOS objective quality scoring (requires --dnsmos-model).")
    p.add_argument("--dnsmos-model", default=None,
                    help="Path to the DNSMOS sig_bak_ovr.onnx model file.")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args(argv)


def build_config(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        limit=args.limit,
        force=args.force,
        language=None if args.language.lower() == "auto" else args.language,
        whisper_model=args.model,
        device=args.device,
        enable_speaker_diarization=args.enable_speaker_diarization,
        hf_token=args.hf_token,
        loudness_normalize=not args.no_loudness_normalize,
        enable_dnsmos=args.enable_dnsmos,
        dnsmos_model_path=args.dnsmos_model,
        log_level=args.log_level,
    )


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# --------------------------------------------------------------------------- #
# Per-file pipeline
# --------------------------------------------------------------------------- #

def discover_audio_files(input_dir: Path) -> List[Path]:
    files = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    )
    return files


def process_file(cfg: PipelineConfig, src_path: Path) -> None:
    source_id = source_id_for(src_path)
    stat = src_path.stat()
    state = load_state(cfg.state_dir, source_id)

    # ---- Determine language for THIS file ----
    # If detect_language_from_filename is on (default), read the language from
    # the filename prefix (hin_1.wav -> Hindi -> whisper "hi"). This is the fix
    # for every non-English file previously being forced to transcribe as
    # English. Falls back to cfg.language (or auto-detect) if the prefix is
    # unrecognized.
    from pipeline.languages import language_from_filename
    file_language = cfg.language  # default/fallback
    lang_info = None
    if cfg.detect_language_from_filename:
        lang_info = language_from_filename(src_path)
        if lang_info.recognized:
            file_language = lang_info.whisper
            logger.info("[%s] language from filename: %s (whisper=%s)",
                        source_id, lang_info.name, lang_info.whisper)
        else:
            logger.warning("[%s] filename prefix '%s' not recognized; "
                           "falling back to language=%s. Add it to pipeline/languages.py if needed.",
                           source_id, lang_info.code, file_language)

    if state is not None and not cfg.force:
        if state.status == "completed" and not is_stale(state, stat.st_size, stat.st_mtime):
            logger.info("[%s] already completed, skipping (use --force to reprocess).", source_id)
            return
        if is_stale(state, stat.st_size, stat.st_mtime):
            logger.info("[%s] source file changed since last run; reprocessing.", source_id)
            state = None

    if state is None or cfg.force:
        state = FileState(
            source_path=str(src_path), source_size=stat.st_size, source_mtime=stat.st_mtime,
            source_id=source_id, status="in_progress",
        )
        if cfg.force:
            # Full redo: drop stale manifest entries for this source so a
            # reprocess doesn't create duplicate rows alongside the old ones.
            for name in ("accepted.jsonl", "quarantined.jsonl", "rejected.jsonl"):
                _remove_records_for_source(cfg.manifests_dir / name, source_id)

    save_state(cfg.state_dir, state)

    source_cache_dir = cfg.cache_dir / source_id
    source_cache_dir.mkdir(parents=True, exist_ok=True)
    analysis_wav = source_cache_dir / "analysis_16k.wav"
    metrics_path = source_cache_dir / "streaming_metrics.json"
    transcript_cache_path = source_cache_dir / "transcription.json"

    # ---- Stage 1: integrity (protocol 1) ----
    logger.info("[%s] checking source integrity...", source_id)
    meta = audio_utils.ffprobe_metadata(src_path)
    audio_utils.verify_integrity(src_path, meta)
    state.mark_stage("integrity", **dataclasses.asdict(meta))
    save_state(cfg.state_dir, state)
    logger.info("[%s] OK: %.1f min, %d ch, %d Hz, codec=%s", source_id,
                meta.duration_sec / 60.0, meta.channels, meta.sample_rate, meta.codec)

    # ---- Stage 2: 16kHz analysis copy (protocol 3) ----
    if cfg.force and analysis_wav.exists():
        analysis_wav.unlink()
    if not analysis_wav.exists():
        logger.info("[%s] creating 16kHz mono analysis copy...", source_id)
        audio_utils.make_analysis_copy(src_path, analysis_wav, cfg.analysis_sr)
    else:
        logger.info("[%s] reusing cached analysis copy.", source_id)
    state.mark_stage("analysis_copy", path=str(analysis_wav))
    save_state(cfg.state_dir, state)

    # ---- Stage 3: streaming quality metrics (protocol 5) ----
    metrics = audio_utils.load_streaming_metrics(metrics_path) if not cfg.force else None
    if metrics is None:
        logger.info("[%s] computing streaming acoustic metrics (single pass)...", source_id)
        metrics = audio_utils.compute_streaming_metrics(
            analysis_wav,
            block_seconds=cfg.metrics_block_seconds,
            silence_dbfs_threshold=cfg.silence_dbfs_threshold,
            clip_amplitude_threshold=cfg.clip_amplitude_threshold,
        )
        audio_utils.save_streaming_metrics(metrics, metrics_path)
    else:
        logger.info("[%s] reusing cached streaming metrics.", source_id)
    logger.info("[%s] rms=%.1f dBFS peak=%.1f dBFS clipped=%.4f%% silence=%.1f%%",
                source_id, metrics.rms_dbfs, metrics.peak_dbfs,
                metrics.clipped_sample_ratio * 100, metrics.silence_ratio * 100)
    state.mark_stage("streaming_metrics", path=str(metrics_path))
    save_state(cfg.state_dir, state)

    # ---- Stage 4+5+9: VAD + transcription (protocols 4, 9) ----
    transcription = transcribe_mod.load_transcription(transcript_cache_path) if not cfg.force else None
    if transcription is None:
        device, compute_type = transcribe_mod.resolve_device_and_compute_type(
            cfg.device, cfg.whisper_compute_type_cpu, cfg.whisper_compute_type_gpu
        )
        logger.info("[%s] transcribing (model=%s device=%s compute_type=%s)... this is the slow step for long files.",
                    source_id, cfg.whisper_model, device, compute_type)
        transcription = transcribe_mod.transcribe_file(
            analysis_wav,
            model_name=cfg.whisper_model,
            device=device,
            compute_type=compute_type,
            language=file_language,
            vad_min_silence_ms=cfg.vad_min_silence_ms,
            vad_speech_pad_ms=cfg.vad_speech_pad_ms,
            beam_size=cfg.beam_size,
        )
        transcribe_mod.save_transcription(transcription, transcript_cache_path)
    else:
        logger.info("[%s] reusing cached transcription (%d segments).", source_id, len(transcription.segments))
    state.mark_stage("transcription", path=str(transcript_cache_path), n_segments=len(transcription.segments))
    save_state(cfg.state_dir, state)

    # File-level noise-floor estimate, reusing the streaming-metrics block trace.
    noise_floor_db, speech_level_db = bg_mod.estimate_noise_floor_from_block_rms(
        metrics.block_rms_dbfs, transcription.vad_speech_regions, metrics.block_hop_sec,
    )
    logger.info("[%s] file-level est. noise floor=%.1f dBFS, speech level=%.1f dBFS (SNR~%.1f dB)",
                source_id, noise_floor_db, speech_level_db, speech_level_db - noise_floor_db)

    # ---- Stage 6: context-aware segmentation (protocols 11, 12, 16) ----
    utterances, seg_notes = seg_mod.build_utterances(
        transcription,
        min_clip_sec=cfg.min_clip_sec,
        preferred_max_sec=cfg.max_clip_sec_preferred,
        hard_cap_sec=cfg.max_clip_sec_hard,
        max_silence_between_merge_sec=cfg.max_silence_between_merge_sec,
        max_internal_pause_sec=cfg.max_internal_pause_sec,
        trimmed_pause_target_sec=cfg.trimmed_pause_target_sec,
    )
    for note in seg_notes:
        logger.warning("[%s] segmentation note: %s", source_id, note)
    logger.info("[%s] %d candidate utterances after segmentation.", source_id, len(utterances))

    accepted_manifest = cfg.manifests_dir / "accepted.jsonl"
    quarantined_manifest = cfg.manifests_dir / "quarantined.jsonl"
    rejected_manifest = cfg.manifests_dir / "rejected.jsonl"

    n_accepted = n_quarantined = n_rejected = 0

    for utt in utterances:
        record = _process_utterance(cfg, src_path, source_id, transcription, utt, noise_floor_db, file_language)
        if record.status == "accepted":
            manifest_mod.append_record(record, accepted_manifest)
            n_accepted += 1
        elif record.status == "quarantined":
            manifest_mod.append_record(record, quarantined_manifest)
            n_quarantined += 1
        else:
            manifest_mod.append_record(record, rejected_manifest)
            n_rejected += 1

    logger.info("[%s] done: %d accepted, %d quarantined, %d rejected.",
                source_id, n_accepted, n_quarantined, n_rejected)

    state.status = "completed"
    state.mark_stage("clips", accepted=n_accepted, quarantined=n_quarantined, rejected=n_rejected)
    save_state(cfg.state_dir, state)


def _remove_records_for_source(manifest_path: Path, source_id: str) -> None:
    if not manifest_path.exists():
        return
    records = manifest_mod.load_records(manifest_path)
    kept = [r for r in records if r.get("source_id") != source_id]
    if len(kept) != len(records):
        manifest_mod.rewrite_records(kept, manifest_path)


def _process_utterance(
    cfg: PipelineConfig,
    src_path: Path,
    source_id: str,
    transcription: transcribe_mod.TranscriptionResult,
    utt: seg_mod.Utterance,
    file_noise_floor_dbfs: float,
    file_language: str = None,
) -> manifest_mod.ClipRecord:
    utterance_id = manifest_mod.new_utterance_id(source_id)
    reject_reasons: List[str] = []
    review_reasons: List[str] = []

    # protocol 10: hallucination detection
    halluc = halluc_mod.check_utterance(
        utt, cfg.whisper_no_speech_prob_threshold,
        cfg.whisper_avg_logprob_threshold, cfg.whisper_compression_ratio_threshold,
    )
    reject_reasons.extend(f"hallucination:{r}" for r in halluc.reject_reasons)
    review_reasons.extend(f"hallucination:{r}" for r in halluc.review_reasons)

    # protocol 13: alignment
    align = align_mod.check_alignment(
        utt, cfg.min_mean_word_confidence_reject, cfg.min_mean_word_confidence_review,
        cfg.min_chars_per_sec, cfg.max_chars_per_sec,
    )
    reject_reasons.extend(f"alignment:{r}" for r in align.reject_reasons)
    review_reasons.extend(f"alignment:{r}" for r in align.review_reasons)

    # protocol 2/14/15: transcript validation. Compare against THIS file's
    # expected language (from filename), not the global default.
    expected_language = file_language if file_language is not None else cfg.language
    tval = tv_mod.validate_transcript(
        utt.text, transcription.language, expected_language,
        cfg.min_language_probability, transcription.language_probability,
    )
    if not tval.ok:
        reject_reasons.extend(f"transcript:{r}" for r in tval.reasons)

    # protocol 12: hard cap enforcement (belt-and-suspenders; segmentation
    # should already guarantee this, but this is the actual gate that
    # decides accept/reject if it's ever violated).
    if utt.total_kept_duration > cfg.max_clip_sec_hard + 0.05:
        reject_reasons.append(f"exceeds_hard_cap:{utt.total_kept_duration:.2f}s")
    if utt.total_kept_duration < cfg.min_clip_sec - 0.05:
        reject_reasons.append(f"below_min_clip_duration:{utt.total_kept_duration:.2f}s")

    # If already rejecting on transcript/hallucination grounds, skip the
    # (comparatively expensive) audio cut + background analysis - no point
    # analyzing audio for a clip we've already decided not to keep.
    samples = None
    sr = None
    bg = None
    dnsmos_sig = dnsmos_bak = dnsmos_ovrl = None
    speaker_result = None
    loud_result = loud_mod.LoudnessResult(applied=False, input_lufs=None, output_lufs=None, gain_db=0.0, note="not run")

    if not reject_reasons:
        try:
            samples, sr = audio_utils.cut_clip_from_spans(src_path, utt.kept_spans, out_path=None)
        except audio_utils.AudioIntegrityError as e:
            reject_reasons.append(f"cut_failed:{e}")

    if samples is not None:
        mono = audio_utils.to_mono(samples)

        # protocol 5: clipping/distortion gate, computed on this clip's own
        # audio (cheap - clips are short). Checked before background/noise
        # analysis since severe clipping makes those readings meaningless.
        clipped_ratio = audio_utils.compute_clip_clipping_ratio(mono, cfg.clip_amplitude_threshold)
        if clipped_ratio >= cfg.clipping_reject_ratio:
            reject_reasons.append(f"clipping:{clipped_ratio*100:.3f}pct_samples_clipped_gte_{cfg.clipping_reject_ratio*100:.3f}pct")
        elif clipped_ratio >= cfg.clipping_review_ratio:
            review_reasons.append(f"clipping:{clipped_ratio*100:.3f}pct_samples_clipped_gte_{cfg.clipping_review_ratio*100:.3f}pct")

        bg = bg_mod.analyze_clip_background(
            mono, sr, cfg.snr_reject_db, cfg.snr_review_db, cfg.flatness_music_threshold,
            file_noise_floor_dbfs=file_noise_floor_dbfs,
        )
        if cfg.background_heuristic_gates:
            # Only gate on the (fragile) background heuristic if explicitly
            # asked to. Default is advisory-only -- see config docstring.
            if bg.verdict == "reject":
                reject_reasons.extend(f"background:{r}" for r in bg.reasons)
            elif bg.verdict == "review":
                review_reasons.extend(f"background:{r}" for r in bg.reasons)

        # DNSMOS: the real, industry-standard quality gate (when enabled).
        # Language-independent, so one threshold works across all 8 languages.
        if cfg.enable_dnsmos:
            from pipeline import dnsmos as dnsmos_mod
            from pathlib import Path as _Path
            dm = dnsmos_mod.score_clip(
                mono, sr,
                _Path(cfg.dnsmos_model_path) if cfg.dnsmos_model_path else None,
            )
            if dm.available:
                dnsmos_sig, dnsmos_bak, dnsmos_ovrl = dm.sig, dm.bak, dm.ovrl
                if dm.ovrl < cfg.dnsmos_reject_ovrl:
                    reject_reasons.append(f"dnsmos:overall_mos_{dm.ovrl:.2f}_below_reject_{cfg.dnsmos_reject_ovrl}")
                elif dm.ovrl < cfg.dnsmos_review_ovrl:
                    review_reasons.append(f"dnsmos:overall_mos_{dm.ovrl:.2f}_below_review_{cfg.dnsmos_review_ovrl}")
            else:
                logger.debug("DNSMOS unavailable for a clip: %s", dm.note)
    else:
        clipped_ratio = 0.0

    # protocol 8: optional speaker verification. Only worth running once the
    # clip has survived every other check, since it's the most expensive
    # per-clip step when enabled.
    if samples is not None and not reject_reasons and cfg.enable_speaker_diarization:
        tmp_clip_path = cfg.cache_dir / "_speaker_check_tmp" / f"{utterance_id}.wav"
        tmp_clip_path.parent.mkdir(parents=True, exist_ok=True)
        import soundfile as sf
        sf.write(str(tmp_clip_path), samples, sr, subtype="PCM_16")
        speaker_result = speaker_mod.verify_single_speaker(
            tmp_clip_path, cfg.enable_speaker_diarization, cfg.hf_token,
        )
        tmp_clip_path.unlink(missing_ok=True)
        if speaker_result.method == "pyannote" and not speaker_result.verified:
            reject_reasons.append(f"speaker:multiple_speakers_detected_{speaker_result.num_speakers_detected}")
    else:
        speaker_result = speaker_mod.SpeakerVerificationResult(verified=False, method="not_run")
        if not reject_reasons and cfg.require_speaker_check_for_acceptance:
            review_reasons.append("speaker:verification_not_run_unverified_single_speaker")

    if tval.needs_number_review and not reject_reasons:
        review_reasons.append("transcript:contains_digits_needs_manual_number_normalization_review")

    status = "accepted"
    if reject_reasons:
        status = "rejected"
    elif review_reasons:
        status = "quarantined"

    audio_path_str: Optional[str] = None
    if status in ("accepted", "quarantined") and samples is not None:
        mono_for_loud = audio_utils.to_mono(samples) if cfg.loudness_normalize else None
        out_dir = cfg.clips_dir if status == "accepted" else (cfg.quarantine_dir / "clips")
        out_path = out_dir / f"{utterance_id}.wav"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if status == "accepted" and cfg.loudness_normalize and mono_for_loud is not None:
            normalized, loud_result = loud_mod.normalize(
                mono_for_loud, sr, cfg.target_lufs, cfg.max_loudness_gain_db,
            )
            import soundfile as sf
            sf.write(str(out_path), normalized, sr, subtype="PCM_16")
        else:
            import soundfile as sf
            sf.write(str(out_path), samples, sr, subtype="PCM_16")

        audio_path_str = str(out_path)

        # Write a transcript alongside the audio for BOTH accepted and
        # quarantined clips - quarantined clips exist specifically so a
        # human can review them, and reviewing audio without its transcript
        # text next to it defeats the point.
        transcript_dir = cfg.transcripts_dir if status == "accepted" else (cfg.quarantine_dir / "transcripts")
        transcript_path = transcript_dir / f"{utterance_id}.txt"
        transcript_path.parent.mkdir(parents=True, exist_ok=True)
        transcript_path.write_text(utt.text, encoding="utf-8")

    return manifest_mod.ClipRecord(
        utterance_id=utterance_id,
        status=status,
        source_id=source_id,
        source_filename=src_path.name,
        start_sec=utt.orig_start,
        end_sec=utt.orig_end,
        duration_sec=utt.total_kept_duration,
        text=utt.text,
        audio_path=audio_path_str,
        detected_language=transcription.language,
        language_probability=transcription.language_probability,
        avg_logprob=utt.avg_logprob,
        no_speech_prob=utt.no_speech_prob,
        compression_ratio=utt.compression_ratio,
        mean_word_confidence=align.mean_word_confidence,
        chars_per_sec=align.chars_per_sec,
        est_snr_db=bg.est_snr_db if bg else -999.0,
        background_verdict=bg.verdict if bg else "not_run",
        clipped_sample_ratio=clipped_ratio,
        dnsmos_sig=dnsmos_sig,
        dnsmos_bak=dnsmos_bak,
        dnsmos_ovrl=dnsmos_ovrl,
        speaker_verification_method=speaker_result.method if speaker_result else "not_run",
        speaker_verified=speaker_result.verified if speaker_result else False,
        loudness_applied=loud_result.applied,
        input_lufs=loud_result.input_lufs,
        output_lufs=loud_result.output_lufs,
        internal_pauses_trimmed=utt.internal_pauses_trimmed,
        needs_human_review=(status == "quarantined"),
        needs_number_review=tval.needs_number_review,
        rejection_reasons=reject_reasons,
        review_reasons=review_reasons,
    )


# --------------------------------------------------------------------------- #
# Dataset-level post-processing: dedup + reports (protocols 18, 20, 21, 22)
# --------------------------------------------------------------------------- #

def run_dataset_postprocessing(cfg: PipelineConfig) -> None:
    accepted_manifest = cfg.manifests_dir / "accepted.jsonl"
    quarantined_manifest = cfg.manifests_dir / "quarantined.jsonl"
    rejected_manifest = cfg.manifests_dir / "rejected.jsonl"

    accepted = manifest_mod.load_records(accepted_manifest)
    quarantined = manifest_mod.load_records(quarantined_manifest)
    rejected = manifest_mod.load_records(rejected_manifest)

    if accepted:
        logger.info("Running dataset-level deduplication over %d accepted clips...", len(accepted))
        candidates = [
            dedup_mod.DedupCandidate(r["utterance_id"], r["text"], r["duration_sec"], r["audio_path"])
            for r in accepted if r.get("audio_path")
        ]
        groups = dedup_mod.find_duplicates(
            candidates, cfg.dedup_text_similarity_threshold,
            cfg.dedup_audio_similarity_threshold, cfg.dedup_duration_tolerance_sec,
        )
        dup_ids_to_demote = set()
        by_id = {r["utterance_id"]: r for r in accepted}
        for group in groups:
            canonical, *dupes = group
            for dup_id in dupes:
                dup_ids_to_demote.add(dup_id)
                rec = by_id[dup_id]
                rec["status"] = "rejected"
                rec["rejection_reasons"] = rec.get("rejection_reasons", []) + [f"duplicate_of:{canonical}"]
                old_path = Path(rec["audio_path"]) if rec.get("audio_path") else None
                if old_path and old_path.exists():
                    dup_dir = cfg.quarantine_dir / "duplicates"
                    dup_dir.mkdir(parents=True, exist_ok=True)
                    new_path = dup_dir / old_path.name
                    old_path.replace(new_path)
                    rec["audio_path"] = str(new_path)
                rejected.append(rec)

        if dup_ids_to_demote:
            logger.info("Deduplication demoted %d clip(s) as duplicates.", len(dup_ids_to_demote))
            accepted = [r for r in accepted if r["utterance_id"] not in dup_ids_to_demote]
            manifest_mod.rewrite_records(accepted, accepted_manifest)
            manifest_mod.rewrite_records(rejected, rejected_manifest)

    logger.info("Writing dataset-level reports...")
    manifest_mod.write_dataset_summary(cfg.reports_dir, accepted, quarantined, rejected)
    manifest_mod.phoneme_coverage_report(cfg.reports_dir, accepted)
    manifest_mod.split_dataset(
        accepted, cfg.reports_dir, cfg.split_train_ratio, cfg.split_val_ratio, cfg.split_seed,
    )
    logger.info("Reports written to %s", cfg.reports_dir)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    setup_logging(args.log_level)
    cfg = build_config(args)
    cfg.ensure_dirs()

    if cfg.enable_speaker_diarization and not cfg.hf_token:
        logger.error("--enable-speaker-diarization requires --hf-token.")
        return 2

    if not cfg.input_dir.exists():
        logger.error("Input directory does not exist: %s", cfg.input_dir)
        return 2

    files = discover_audio_files(cfg.input_dir)
    if cfg.limit is not None:
        files = files[: cfg.limit]

    if not files:
        logger.warning("No audio files found in %s (looked for extensions: %s)",
                        cfg.input_dir, sorted(AUDIO_EXTENSIONS))
        return 0

    logger.info("Found %d source file(s) to process.", len(files))

    n_ok = n_failed = 0
    t_start = time.time()
    for i, f in enumerate(files, 1):
        logger.info("=== [%d/%d] %s ===", i, len(files), f.name)
        try:
            process_file(cfg, f)
            n_ok += 1
        except audio_utils.AudioIntegrityError as e:
            logger.error("[%s] INTEGRITY FAILURE, quarantining whole file: %s", f.name, e)
            _mark_file_failed(cfg, f, str(e))
            n_failed += 1
        except Exception as e:  # noqa: BLE001 - one bad file must not kill the run
            logger.error("[%s] UNEXPECTED ERROR: %s", f.name, e)
            logger.error(traceback.format_exc())
            _mark_file_failed(cfg, f, f"{type(e).__name__}: {e}")
            n_failed += 1

    run_dataset_postprocessing(cfg)

    elapsed = time.time() - t_start
    logger.info("=== Run complete in %.1f min: %d file(s) OK, %d failed. ===",
                elapsed / 60.0, n_ok, n_failed)
    return 0 if n_failed == 0 else 1


def _mark_file_failed(cfg: PipelineConfig, path: Path, error: str) -> None:
    source_id = source_id_for(path)
    try:
        stat = path.stat()
        state = load_state(cfg.state_dir, source_id) or FileState(
            source_path=str(path), source_size=stat.st_size, source_mtime=stat.st_mtime, source_id=source_id,
        )
        state.status = "failed"
        state.error = error
        save_state(cfg.state_dir, state)
    except Exception:
        logger.error("Additionally failed to persist failure state for %s", path)


if __name__ == "__main__":
    sys.exit(main())
