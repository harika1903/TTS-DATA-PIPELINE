#!/usr/bin/env python3
"""TTS pipeline entry point using the TEAM'S CUSTOM STT model (approach B).

This is a SEPARATE entry point from tts_pipeline.py (which uses Whisper).
Use this one when you want the custom STT server instead of Whisper:

    python tts_pipeline_custom.py --input Audios --output results --limit 1 \
        --stt-url http://0.0.0.0:8123 --enable-dnsmos --dnsmos-model dnsmos_model.onnx

HOW IT DIFFERS FROM THE WHISPER PIPELINE:
  * No Whisper. Clips are cut at SILENCE (silence_segmentation.py), because the
    custom STT server returns text with no timestamps.
  * Each cut clip is sent to the custom STT server for its transcript.
  * All the audio-quality gates still apply: clipping, DNSMOS, dedup, manifest.
  * Some Whisper-only checks are NOT available here and are therefore skipped:
      - word-level confidence (no per-word data from the server)
      - hallucination signals based on Whisper's no_speech_prob/logprob
      - precise word-boundary alignment
    This is the honest tradeoff of not having timestamps.

READ THIS: the transcript quality now depends entirely on the custom STT
server. Verify its output per language (a native speaker should spot-check),
exactly as you would with Whisper.
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
from pipeline import silence_segmentation as sil
from pipeline import custom_stt
from pipeline.languages import language_from_filename
from pipeline import background as bg_mod
from pipeline import dedup as dedup_mod
from pipeline import manifest as manifest_mod
from pipeline import transcript_validate as tv_mod
from pipeline import loudness as loud_mod

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".wma", ".opus"}
logger = logging.getLogger("tts_pipeline_custom")


def _lang_display():
    from pipeline.languages import LANGUAGES
    return LANGUAGES


def _make_forced_lang_info(stt_lang, display_name):
    """Build a LanguageInfo-like object for a forced language, so the rest of
    the pipeline (which reads lang_info.name / .whisper) works unchanged."""
    from pipeline.languages import LanguageInfo, CUSTOM_STT_LANG, LANGUAGES
    # find the code whose custom_stt matches, to fill whisper/name properly
    code = next((c for c, v in CUSTOM_STT_LANG.items() if v == stt_lang), stt_lang[:3])
    if code in LANGUAGES:
        name, whisper, nemo, hunspell = LANGUAGES[code]
    else:
        name, whisper, nemo, hunspell = display_name, None, None, None
    return LanguageInfo(code=code, name=name, whisper=whisper, nemo=nemo,
                        hunspell=hunspell, custom_stt=stt_lang, recognized=True)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TTS dataset pipeline using the team's custom STT model (silence-based cutting).")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--stt-url", default="http://0.0.0.0:8123", help="Base URL of the custom STT server.")
    p.add_argument("--language", default=None,
                    help="Force ALL files in this run to this language (e.g. hindi, telugu, kannada). "
                         "Use this to process one language at a time -- no filename prefix needed. "
                         "Supported: hindi, telugu, kannada, tamil, malayalam, bengali, gujarati, english.")
    p.add_argument("--stt-timeout", type=float, default=120.0, help="Per-clip STT request timeout (seconds).")
    p.add_argument("--silence-db", type=float, default=-38.0, help="Silence threshold in dBFS (raise toward -30 for noisy audio).")
    p.add_argument("--min-silence", type=float, default=0.20, help="Minimum pause length (sec) that counts as a cut point.")
    p.add_argument("--end-pad", type=float, default=0.40, help="Extra audio (sec) at clip end to capture trailing words. Default 0.40.")
    p.add_argument("--enable-dnsmos", action="store_true")
    p.add_argument("--dnsmos-model", default=None)
    p.add_argument("--no-bak-gate", action="store_true",
                    help="Disable the BAK background-music/noise gate (on by default when DNSMOS is enabled).")
    p.add_argument("--bak-reject", type=float, default=3.0,
                    help="Reject clips with DNSMOS background score below this (music/noise). Default 2.5.")
    p.add_argument("--bak-review", type=float, default=4.0,
                    help="Review clips with DNSMOS background score below this. Default 3.0.")
    p.add_argument("--no-loudness-normalize", action="store_true")
    p.add_argument("--flag-numbers", action="store_true",
                    help="Quarantine clips containing digits for manual number review (off by default).")
    p.add_argument("--translit-numbers", action="store_true",
                    help="Transliterate digits to spoken number words in the target script "
                         "(17 -> సెవెంటీన్). Telugu only for now; other languages pass through unchanged.")
    p.add_argument("--translit-style", default="english", choices=["english", "native"],
                    help="Number transliteration style (default english: 21 -> ట్వెంటీ వన్).")
    p.add_argument("--enable-speaker-analysis", action="store_true",
                    help="Enable pyannote one-speaker-per-file + cross-file speaker dedup (needs --hf-token).")
    p.add_argument("--hf-token", default=None, help="Hugging Face token for pyannote.")
    p.add_argument("--speaker-device", default="cuda", choices=["cuda", "cpu"], help="Device for pyannote.")
    p.add_argument("--dominant-fraction", type=float, default=0.85,
                    help="If the top speaker is at least this fraction of speech, keep the file as "
                         "single-speaker and drop only clips overlapping a second voice. Default 0.90. "
                         "Set to 1.0 for strict (any 2nd speaker rejects the whole file).")
    p.add_argument("--speaker-dup-reject", type=float, default=0.75,
                    help="Voiceprint cosine similarity at/above which a file is a duplicate speaker (rejected).")
    p.add_argument("--speaker-dup-review", type=float, default=0.60,
                    help="Voiceprint similarity at/above which a borderline match is flagged for review.")
    p.add_argument("--speaker-db", default=None,
                    help="Path to the GLOBAL speaker database (shared across all runs/folders). "
                         "Default: speaker_db_global.json next to the pipeline. Use the SAME path for all "
                         "runs so a speaker is deduplicated across every language folder.")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args(argv)


def build_config(args) -> PipelineConfig:
    return PipelineConfig(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        limit=args.limit,
        force=args.force,
        use_custom_stt=True,
        custom_stt_url=args.stt_url,
        force_language=args.language,
        custom_stt_timeout_sec=args.stt_timeout,
        silence_db_threshold=args.silence_db,
        min_silence_sec=args.min_silence,
        clip_end_pad_sec=args.end_pad,
        enable_dnsmos=args.enable_dnsmos,
        dnsmos_model_path=args.dnsmos_model,
        enable_bak_gate=not args.no_bak_gate,
        dnsmos_reject_bak=args.bak_reject,
        dnsmos_review_bak=args.bak_review,
        loudness_normalize=not args.no_loudness_normalize,
        flag_numbers_for_review=args.flag_numbers,
        enable_number_translit=args.translit_numbers,
        number_translit_style=args.translit_style,
        enable_speaker_analysis=args.enable_speaker_analysis,
        hf_token=args.hf_token,
        speaker_device=args.speaker_device,
        dominant_speaker_min_fraction=args.dominant_fraction,
        speaker_dup_reject_similarity=args.speaker_dup_reject,
        speaker_dup_review_similarity=args.speaker_dup_review,
        speaker_db_path=args.speaker_db,
        log_level=args.log_level,
    )


def setup_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level),
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")


def discover_audio_files(input_dir: Path) -> List[Path]:
    return sorted(p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS)


def process_file(cfg: PipelineConfig, src_path: Path, progress_callback=None) -> None:
    source_id = source_id_for(src_path)
    stat = src_path.stat()

    lang_info = language_from_filename(src_path)
    if cfg.force_language:
        # A language was forced for the whole run (per-language processing):
        # every file in this folder is treated as this language, regardless of
        # its filename. This is how you "run a pipeline per language" -- sort
        # files into per-language folders and force the language for each run.
        stt_language = cfg.force_language
        # keep a display name from the map if we recognize it
        from pipeline.languages import CUSTOM_STT_LANG
        disp = next((info[0] for code, info in _lang_display().items()
                     if CUSTOM_STT_LANG.get(code) == cfg.force_language), cfg.force_language.title())
        lang_info = _make_forced_lang_info(cfg.force_language, disp)
        logger.info("[%s] language FORCED for this run: %s (STT lang='%s')", source_id, disp, stt_language)
    elif lang_info.recognized and lang_info.custom_stt:
        stt_language = lang_info.custom_stt
        logger.info("[%s] language from filename: %s (STT lang='%s')", source_id, lang_info.name, stt_language)
    else:
        stt_language = "english"
        logger.warning("[%s] prefix '%s' not recognized and no --language set; defaulting to 'english'. "
                        "Either use --language <lang> for the whole folder, or name files with a "
                        "language prefix.", source_id, lang_info.code)

    state = load_state(cfg.state_dir, source_id)
    if state is not None and not cfg.force:
        if state.status == "completed" and not is_stale(state, stat.st_size, stat.st_mtime):
            logger.info("[%s] already completed, skipping (use --force).", source_id)
            return
    state = FileState(source_path=str(src_path), source_size=stat.st_size,
                      source_mtime=stat.st_mtime, source_id=source_id, status="in_progress")
    if cfg.force:
        for name in ("accepted.jsonl", "quarantined.jsonl", "rejected.jsonl"):
            _remove_records_for_source(cfg.manifests_dir / name, source_id)
    save_state(cfg.state_dir, state)

    source_cache_dir = cfg.cache_dir / source_id
    source_cache_dir.mkdir(parents=True, exist_ok=True)
    analysis_wav = source_cache_dir / "analysis_16k.wav"

    # ---- integrity ----
    logger.info("[%s] checking source integrity...", source_id)
    meta = audio_utils.ffprobe_metadata(src_path)
    audio_utils.verify_integrity(src_path, meta)
    logger.info("[%s] OK: %.1f min, %d ch, %d Hz", source_id, meta.duration_sec / 60.0, meta.channels, meta.sample_rate)

    # ---- 16k analysis copy (used for silence detection) ----
    if cfg.force and analysis_wav.exists():
        analysis_wav.unlink()
    if not analysis_wav.exists():
        logger.info("[%s] creating 16kHz mono analysis copy...", source_id)
        audio_utils.make_analysis_copy(src_path, analysis_wav, cfg.analysis_sr)
    else:
        logger.info("[%s] reusing cached analysis copy.", source_id)

    # ---- streaming metrics + file-level noise floor ----
    metrics = audio_utils.compute_streaming_metrics(
        analysis_wav, block_seconds=cfg.metrics_block_seconds,
        silence_dbfs_threshold=cfg.silence_dbfs_threshold,
        clip_amplitude_threshold=cfg.clip_amplitude_threshold,
    )
    logger.info("[%s] rms=%.1f dBFS peak=%.1f dBFS clipped=%.4f%% silence=%.1f%%",
                source_id, metrics.rms_dbfs, metrics.peak_dbfs,
                metrics.clipped_sample_ratio * 100, metrics.silence_ratio * 100)
    # file-level noise floor (reuse block trace); no VAD speech regions here, so
    # pass an empty list -> it uses the low percentile of all blocks.
    file_noise_floor_db, _ = bg_mod.estimate_noise_floor_from_block_rms(
        metrics.block_rms_dbfs, [], metrics.block_hop_sec,
    )

    # ---- FILE-LEVEL SPEAKER CHECKS (pyannote) — run BEFORE clip-cutting ----
    # Two checks: (1) one speaker per file, (2) that speaker not already in the
    # dataset. If either fails, reject the whole file now and skip clip-cutting
    # entirely (no point processing a file we're going to discard).
    if cfg.enable_speaker_analysis:
        from pipeline import speaker_analysis as spk
        logger.info("[%s] running speaker analysis (pyannote)...", source_id)
        spk_result = spk.analyze_file_speakers(analysis_wav, cfg.hf_token, cfg.speaker_device)

        if not spk_result.ran:
            # pyannote couldn't run -> don't silently accept. Reject the file
            # with a clear reason so it's visible (safer than assuming 1 speaker).
            logger.warning("[%s] speaker analysis did not run (%s); rejecting file for review.",
                            source_id, spk_result.error)
            _reject_whole_file(cfg, src_path, source_id, meta,
                                f"speaker_analysis_failed:{spk_result.error}")
            state.status = "completed"
            save_state(cfg.state_dir, state)
            return

        logger.info("[%s] detected %d speaker(s) in file (dominant speaker = %.1f%% of speech).",
                    source_id, spk_result.num_speakers, spk_result.dominant_fraction * 100)

        # Check 1: one-speaker-per-file, with dominant-speaker tolerance.
        # If the file has multiple speakers BUT one dominates (>= threshold),
        # treat it as effectively single-speaker: accept the file and drop only
        # the clips that overlap the minority speaker(s). Only reject the whole
        # file if it's genuinely mixed (no dominant speaker).
        minority_segments = []
        if spk_result.num_speakers > cfg.max_speakers_per_file:
            if spk_result.dominant_fraction >= cfg.dominant_speaker_min_fraction:
                minority_segments = spk_result.minority_segments
                logger.info("[%s] %d speakers, but dominant speaker is %.1f%% (>= %.0f%%) — "
                            "keeping file; will drop %d clip-region(s) with a second voice.",
                            source_id, spk_result.num_speakers, spk_result.dominant_fraction * 100,
                            cfg.dominant_speaker_min_fraction * 100, len(minority_segments))
            else:
                logger.info("[%s] REJECTED whole file: %d speakers, none dominant "
                            "(top speaker only %.1f%% < %.0f%%).",
                            source_id, spk_result.num_speakers, spk_result.dominant_fraction * 100,
                            cfg.dominant_speaker_min_fraction * 100)
                _reject_whole_file(cfg, src_path, source_id, meta,
                                    f"multiple_speakers_no_dominant:{spk_result.num_speakers}_top_{spk_result.dominant_fraction:.2f}")
                state.status = "completed"
                save_state(cfg.state_dir, state)
                return

        # Check 2: cross-file speaker dedup, using the GLOBAL database so
        # speakers are deduplicated across ALL output folders / languages.
        from pathlib import Path as _P
        if cfg.speaker_db_path:
            global_db_path = _P(cfg.speaker_db_path)
        else:
            # Fixed default: next to this script, shared by every run.
            global_db_path = _P(__file__).resolve().parent / "speaker_db_global.json"
        db = spk.SpeakerDatabase(
            global_db_path,
            cfg.speaker_dup_reject_similarity,
            cfg.speaker_dup_review_similarity,
        )
        match = db.check(spk_result.embedding)
        if match.decision == "duplicate":
            logger.info("[%s] REJECTED whole file: speaker already in dataset "
                        "(matches %s, similarity %.2f).", source_id, match.matched_source_id, match.best_similarity)
            _reject_whole_file(cfg, src_path, source_id, meta,
                                f"speaker_already_in_dataset:matches_{match.matched_source_id}_sim_{match.best_similarity:.2f}")
            state.status = "completed"
            save_state(cfg.state_dir, state)
            return
        elif match.decision == "review":
            logger.warning("[%s] speaker BORDERLINE match to %s (similarity %.2f) — "
                            "processing file but flagging clips for review.",
                            source_id, match.matched_source_id, match.best_similarity)
            # We still process it, but mark it; the clips will carry a review note.
            _file_speaker_review_note = f"speaker_borderline_match:{match.matched_source_id}_sim_{match.best_similarity:.2f}"
        else:
            _file_speaker_review_note = None
            logger.info("[%s] new speaker (best similarity to existing: %.2f) — accepted, added to DB.",
                        source_id, match.best_similarity)

        # Register this file's speaker in the DB (so future files dedup against it).
        db.add(source_id, spk_result.embedding)
    else:
        _file_speaker_review_note = None
        minority_segments = []

    # ---- silence-based cut points (over the whole analysis copy, streamed) ----
    logger.info("[%s] detecting silence cut points (threshold=%.0f dBFS, min pause=%.2fs)...",
                source_id, cfg.silence_db_threshold, cfg.min_silence_sec)
    # Read the (small) analysis copy fully to compute the energy envelope.
    import soundfile as sf
    analysis_samples, analysis_sr = sf.read(str(analysis_wav), dtype="float32", always_2d=False)
    cut_points = sil.find_cut_points(
        analysis_samples, analysis_sr,
        silence_db_threshold=cfg.silence_db_threshold,
        min_silence_sec=cfg.min_silence_sec,
    )
    clips, seg_notes = sil.build_clips_from_cuts(
        total_duration_sec=meta.duration_sec,
        cut_points=cut_points,
        min_clip_sec=cfg.min_clip_sec,
        preferred_max_sec=cfg.max_clip_sec_preferred,
        hard_cap_sec=cfg.max_clip_sec_hard,
        end_pad_sec=cfg.clip_end_pad_sec,
    )
    for note in seg_notes:
        logger.warning("[%s] segmentation note: %s", source_id, note)
    logger.info("[%s] %d candidate clips from silence segmentation.", source_id, len(clips))

    # free the big array
    del analysis_samples

    accepted_manifest = cfg.manifests_dir / "accepted.jsonl"
    quarantined_manifest = cfg.manifests_dir / "quarantined.jsonl"
    rejected_manifest = cfg.manifests_dir / "rejected.jsonl"
    n_accepted = n_quarantined = n_rejected = 0

    # report total clips to the progress callback (for clip-level progress bar)
    if progress_callback:
        progress_callback(source_id, 0, len(clips))

    for idx, clip in enumerate(clips):
        rec = _process_clip(cfg, src_path, source_id, clip, stt_language,
                            lang_info, file_noise_floor_db, meta, _file_speaker_review_note,
                            minority_segments)
        if rec.status == "accepted":
            manifest_mod.append_record(rec, accepted_manifest); n_accepted += 1
        elif rec.status == "quarantined":
            manifest_mod.append_record(rec, quarantined_manifest); n_quarantined += 1
        else:
            manifest_mod.append_record(rec, rejected_manifest); n_rejected += 1
        if (idx + 1) % 25 == 0:
            logger.info("[%s]   processed %d/%d clips...", source_id, idx + 1, len(clips))
        if progress_callback:
            progress_callback(source_id, idx + 1, len(clips))

    logger.info("[%s] done: %d accepted, %d quarantined, %d rejected.",
                source_id, n_accepted, n_quarantined, n_rejected)
    state.status = "completed"
    state.mark_stage("clips", accepted=n_accepted, quarantined=n_quarantined, rejected=n_rejected)
    save_state(cfg.state_dir, state)


def _process_clip(cfg, src_path, source_id, clip, stt_language, lang_info, file_noise_floor_dbfs, meta, file_speaker_review_note=None, minority_segments=None):
    utterance_id = manifest_mod.new_utterance_id(source_id)
    reject_reasons: List[str] = []
    review_reasons: List[str] = []

    # If this clip overlaps a minority (second) speaker's segment, drop it —
    # this is how a mostly-single-speaker file keeps its good clips while
    # discarding only the parts where a second voice appears.
    if minority_segments:
        from pipeline import speaker_analysis as _spk
        if _spk.clip_overlaps_minority(clip.start_sec, clip.end_sec, minority_segments, cfg.minority_clip_overlap_sec):
            reject_reasons.append("clip_overlaps_second_speaker")

    # Cut the clip audio from the ORIGINAL file (full quality)
    try:
        samples, sr = audio_utils.cut_clip_from_spans(src_path, [(clip.start_sec, clip.end_sec)], out_path=None)
    except audio_utils.AudioIntegrityError as e:
        return _rejected_record(utterance_id, source_id, src_path, clip, f"cut_failed:{e}", stt_language)

    mono = audio_utils.to_mono(samples)

    # duration gate
    if clip.duration > cfg.max_clip_sec_hard + 0.05:
        reject_reasons.append(f"exceeds_hard_cap:{clip.duration:.2f}s")
    if clip.duration < cfg.min_clip_sec - 0.05:
        reject_reasons.append(f"below_min_clip_duration:{clip.duration:.2f}s")

    # clipping gate
    clipped_ratio = audio_utils.compute_clip_clipping_ratio(mono, cfg.clip_amplitude_threshold)
    if clipped_ratio >= cfg.clipping_reject_ratio:
        reject_reasons.append(f"clipping:{clipped_ratio*100:.3f}pct_samples_clipped")
    elif clipped_ratio >= cfg.clipping_review_ratio:
        review_reasons.append(f"clipping:{clipped_ratio*100:.3f}pct_samples_clipped")

    # background heuristic (advisory only, same as Whisper pipeline)
    bg = bg_mod.analyze_clip_background(
        mono, sr, cfg.snr_reject_db, cfg.snr_review_db, cfg.flatness_music_threshold,
        file_noise_floor_dbfs=file_noise_floor_dbfs,
    )

    # DNSMOS quality gate
    dnsmos_sig = dnsmos_bak = dnsmos_ovrl = None
    if cfg.enable_dnsmos and not reject_reasons:
        from pipeline import dnsmos as dnsmos_mod
        dm = dnsmos_mod.score_clip(mono, sr, Path(cfg.dnsmos_model_path) if cfg.dnsmos_model_path else None)
        if dm.available:
            dnsmos_sig, dnsmos_bak, dnsmos_ovrl = dm.sig, dm.bak, dm.ovrl
            if dm.ovrl < cfg.dnsmos_reject_ovrl:
                reject_reasons.append(f"dnsmos:overall_mos_{dm.ovrl:.2f}_below_reject_{cfg.dnsmos_reject_ovrl}")
            elif dm.ovrl < cfg.dnsmos_review_ovrl:
                review_reasons.append(f"dnsmos:overall_mos_{dm.ovrl:.2f}_below_review_{cfg.dnsmos_review_ovrl}")
            # BAK gate (protocol 7): background music/noise/crowd detection.
            # Low BAK = contaminated background, even if speech (SIG) is clean.
            if cfg.enable_bak_gate:
                if dm.bak < cfg.dnsmos_reject_bak:
                    reject_reasons.append(f"background_music_or_noise:bak_{dm.bak:.2f}_below_reject_{cfg.dnsmos_reject_bak}")
                elif dm.bak < cfg.dnsmos_review_bak:
                    review_reasons.append(f"background_music_or_noise:bak_{dm.bak:.2f}_below_review_{cfg.dnsmos_review_bak}")

    # If audio already failed, don't waste an STT call on it.
    text = ""
    if not reject_reasons:
        # Write the clip to a temp file to send to the STT server. Use the
        # MONO version (same as what gets saved). CRUCIALLY, pad with SILENCE
        # on BOTH ends before sending: STT models can drop the first or last
        # word when speech starts/ends right at the audio boundary (they need
        # a little silence on each side to "finalize" edge words). Padding one
        # side only fixes that side and drops the other, so we pad both. The
        # silence is added ONLY for the STT call -- the saved clip (below) has
        # none, so no dead air enters the dataset.
        tmp_dir = cfg.cache_dir / "_stt_tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_clip = tmp_dir / f"{utterance_id}.wav"
        import soundfile as sf
        import numpy as _np
        pad = _np.zeros(int(cfg.stt_trailing_silence_sec * sr), dtype=mono.dtype)
        mono_for_stt = _np.concatenate([pad, mono, pad])  # silence on both ends
        sf.write(str(tmp_clip), mono_for_stt, sr, subtype="PCM_16")

        stt = custom_stt.transcribe_clip(str(tmp_clip), cfg.custom_stt_url, stt_language, cfg.custom_stt_timeout_sec)
        tmp_clip.unlink(missing_ok=True)

        if not stt.ok:
            reject_reasons.append(f"stt_failed:{stt.error}")
        else:
            text = stt.transcript
            # numbers as spoken words in the target script (digits -> words).
            # Runs before validation so the written transcript carries words, not digits.
            if getattr(cfg, "enable_number_translit", False):
                try:
                    from pipeline import number_translit as _nt
                    text = _nt.transliterate_numbers(
                        text, stt_language, getattr(cfg, "number_translit_style", "english")).text
                except Exception as _e:  # never let number handling break a clip
                    logger.warning("[%s] number transliteration skipped: %s", source_id, _e)
            # transcript validation (empty/garbage/language checks)
            tval = tv_mod.validate_transcript(
                text, lang_info.whisper or "unknown", lang_info.whisper,
                cfg.min_language_probability, 1.0,  # no lang-prob from custom STT
            )
            if not tval.ok:
                # empty/garbage transcript is a real reject; language checks here
                # are weak (no lang-prob), so only reject on empty/garbage.
                hard = [r for r in tval.reasons if "empty" in r or "no_alphabetic" in r or "control_char" in r]
                if hard:
                    reject_reasons.extend(f"transcript:{r}" for r in hard)
                else:
                    review_reasons.extend(f"transcript:{r}" for r in tval.reasons)
            if cfg.flag_numbers_for_review and tval.needs_number_review and not reject_reasons:
                review_reasons.append("transcript:contains_digits_needs_manual_number_review")

    # If the file was a borderline cross-file speaker match, flag every clip.
    if file_speaker_review_note:
        review_reasons.append(file_speaker_review_note)

    status = "rejected" if reject_reasons else ("quarantined" if review_reasons else "accepted")

    audio_path_str = None
    loud_result = loud_mod.LoudnessResult(applied=False, input_lufs=None, output_lufs=None, gain_db=0.0, note="not run")
    if status in ("accepted", "quarantined"):
        out_dir = cfg.clips_dir if status == "accepted" else (cfg.quarantine_dir / "clips")
        out_path = out_dir / f"{utterance_id}.wav"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        import soundfile as sf
        if status == "accepted" and cfg.loudness_normalize:
            normalized, loud_result = loud_mod.normalize(mono, sr, cfg.target_lufs, cfg.max_loudness_gain_db)
            sf.write(str(out_path), normalized, sr, subtype="PCM_16")
        else:
            sf.write(str(out_path), samples, sr, subtype="PCM_16")
        audio_path_str = str(out_path)

        tdir = cfg.transcripts_dir if status == "accepted" else (cfg.quarantine_dir / "transcripts")
        tpath = tdir / f"{utterance_id}.txt"
        tpath.parent.mkdir(parents=True, exist_ok=True)
        tpath.write_text(text, encoding="utf-8")

    return manifest_mod.ClipRecord(
        utterance_id=utterance_id, status=status, source_id=source_id, source_filename=src_path.name,
        start_sec=clip.start_sec, end_sec=clip.end_sec, duration_sec=clip.duration, text=text,
        audio_path=audio_path_str, detected_language=lang_info.name, language_probability=1.0,
        avg_logprob=0.0, no_speech_prob=0.0, compression_ratio=0.0, mean_word_confidence=0.0,
        chars_per_sec=(len(text) / clip.duration if clip.duration > 0 else 0.0),
        est_snr_db=bg.est_snr_db if bg else -999.0, background_verdict=bg.verdict if bg else "not_run",
        clipped_sample_ratio=clipped_ratio, speaker_verification_method="not_run", speaker_verified=False,
        loudness_applied=loud_result.applied, input_lufs=loud_result.input_lufs, output_lufs=loud_result.output_lufs,
        internal_pauses_trimmed=0, needs_human_review=(status == "quarantined"),
        needs_number_review=False, rejection_reasons=reject_reasons, review_reasons=review_reasons,
        dnsmos_sig=dnsmos_sig, dnsmos_bak=dnsmos_bak, dnsmos_ovrl=dnsmos_ovrl,
    )


def _rejected_record(utterance_id, source_id, src_path, clip, reason, stt_language):
    return manifest_mod.ClipRecord(
        utterance_id=utterance_id, status="rejected", source_id=source_id, source_filename=src_path.name,
        start_sec=clip.start_sec, end_sec=clip.end_sec, duration_sec=clip.duration, text="",
        audio_path=None, detected_language="", language_probability=1.0, avg_logprob=0.0,
        no_speech_prob=0.0, compression_ratio=0.0, mean_word_confidence=0.0, chars_per_sec=0.0,
        est_snr_db=-999.0, background_verdict="not_run", clipped_sample_ratio=0.0,
        speaker_verification_method="not_run", speaker_verified=False, loudness_applied=False,
        input_lufs=None, output_lufs=None, internal_pauses_trimmed=0, needs_human_review=False,
        needs_number_review=False, rejection_reasons=[reason], review_reasons=[],
    )


def _reject_whole_file(cfg, src_path, source_id, meta, reason):
    """Record a whole-file rejection (multiple speakers, speaker already in
    dataset, or speaker analysis failed). Writes one rejected-manifest row so
    the decision is traceable; no clips are cut.
    """
    rec = manifest_mod.ClipRecord(
        utterance_id=manifest_mod.new_utterance_id(source_id), status="rejected",
        source_id=source_id, source_filename=src_path.name,
        start_sec=0.0, end_sec=meta.duration_sec, duration_sec=meta.duration_sec, text="",
        audio_path=None, detected_language="", language_probability=1.0, avg_logprob=0.0,
        no_speech_prob=0.0, compression_ratio=0.0, mean_word_confidence=0.0, chars_per_sec=0.0,
        est_snr_db=-999.0, background_verdict="not_run", clipped_sample_ratio=0.0,
        speaker_verification_method="pyannote", speaker_verified=False, loudness_applied=False,
        input_lufs=None, output_lufs=None, internal_pauses_trimmed=0, needs_human_review=False,
        needs_number_review=False, rejection_reasons=[f"file_level:{reason}"], review_reasons=[],
    )
    manifest_mod.append_record(rec, cfg.manifests_dir / "rejected.jsonl")


def _remove_records_for_source(manifest_path: Path, source_id: str) -> None:
    if not manifest_path.exists():
        return
    records = manifest_mod.load_records(manifest_path)
    kept = [r for r in records if r.get("source_id") != source_id]
    if len(kept) != len(records):
        manifest_mod.rewrite_records(kept, manifest_path)


def run_dataset_postprocessing(cfg: PipelineConfig) -> None:
    accepted = manifest_mod.load_records(cfg.manifests_dir / "accepted.jsonl")
    quarantined = manifest_mod.load_records(cfg.manifests_dir / "quarantined.jsonl")
    rejected = manifest_mod.load_records(cfg.manifests_dir / "rejected.jsonl")
    if accepted:
        logger.info("Running dataset-level deduplication over %d accepted clips...", len(accepted))
        cands = [dedup_mod.DedupCandidate(r["utterance_id"], r["text"], r["duration_sec"], r["audio_path"])
                 for r in accepted if r.get("audio_path")]
        groups = dedup_mod.find_duplicates(cands, cfg.dedup_text_similarity_threshold,
                                            cfg.dedup_audio_similarity_threshold, cfg.dedup_duration_tolerance_sec)
        demote = set()
        by_id = {r["utterance_id"]: r for r in accepted}
        for group in groups:
            canonical, *dupes = group
            for d in dupes:
                demote.add(d); rec = by_id[d]
                rec["status"] = "rejected"
                rec["rejection_reasons"] = rec.get("rejection_reasons", []) + [f"duplicate_of:{canonical}"]
                rejected.append(rec)
        if demote:
            logger.info("Deduplication demoted %d duplicate clip(s).", len(demote))
            accepted = [r for r in accepted if r["utterance_id"] not in demote]
            manifest_mod.rewrite_records(accepted, cfg.manifests_dir / "accepted.jsonl")
            manifest_mod.rewrite_records(rejected, cfg.manifests_dir / "rejected.jsonl")
    manifest_mod.write_dataset_summary(cfg.reports_dir, accepted, quarantined, rejected)
    manifest_mod.phoneme_coverage_report(cfg.reports_dir, accepted)
    manifest_mod.split_dataset(accepted, cfg.reports_dir, cfg.split_train_ratio, cfg.split_val_ratio, cfg.split_seed)
    _write_timestamps_file(cfg, accepted)
    logger.info("Reports written to %s", cfg.reports_dir)


def _write_timestamps_file(cfg, accepted):
    """Write clip-timestamp mappings recording, for every accepted clip, which
    time range of the ORIGINAL source audio it was cut from.

    Writes TWO files:
      * clip_timestamps.csv  - for Excel / spreadsheets
      * clip_timestamps.txt  - human-readable, grouped by source file, aligned
                                columns; nice to read in VS Code / any editor
    """
    import csv

    def hms(seconds):
        if seconds is None:
            return ""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = seconds % 60
        return f"{h:02d}:{m:02d}:{s:06.3f}"

    rows = sorted(accepted, key=lambda r: (r.get("source_filename", ""), r.get("start_sec", 0)))

    # ---- CSV (for spreadsheets) ----
    csv_path = cfg.reports_dir / "clip_timestamps.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["clip_id", "source_file", "start_sec", "end_sec",
                    "start_hms", "end_hms", "duration_sec", "language", "transcript"])
        for r in rows:
            w.writerow([
                r.get("utterance_id", ""), r.get("source_filename", ""),
                f"{r.get('start_sec', 0):.3f}", f"{r.get('end_sec', 0):.3f}",
                hms(r.get("start_sec")), hms(r.get("end_sec")),
                f"{r.get('duration_sec', 0):.3f}",
                r.get("detected_language", ""), r.get("text", ""),
            ])

    # ---- readable TXT (grouped by file, aligned) ----
    txt_path = cfg.reports_dir / "clip_timestamps.txt"
    from collections import defaultdict, OrderedDict
    by_file = OrderedDict()
    for r in rows:
        by_file.setdefault(r.get("source_filename", "unknown"), []).append(r)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 78 + "\n")
        f.write("CLIP TIMESTAMPS — where each clip was cut from the original audio\n")
        f.write("=" * 78 + "\n")
        total = sum(len(v) for v in by_file.values())
        f.write(f"Total accepted clips: {total}   Source files: {len(by_file)}\n")
        f.write("\n")

        for src, clips in by_file.items():
            lang = clips[0].get("detected_language", "")
            f.write("-" * 78 + "\n")
            f.write(f"FILE: {src}   ({lang})   —   {len(clips)} clips\n")
            f.write("-" * 78 + "\n")
            f.write(f"{'#':>3}  {'START':>12}  {'END':>12}  {'LENGTH':>7}   TRANSCRIPT\n")
            for i, r in enumerate(clips, 1):
                start = hms(r.get("start_sec"))
                end = hms(r.get("end_sec"))
                dur = f"{r.get('duration_sec', 0):.1f}s"
                text = (r.get("text", "") or "").strip()
                # first line: the timing + start of transcript; wrap long transcripts
                f.write(f"{i:>3}  {start:>12}  {end:>12}  {dur:>7}   {text}\n")
            f.write("\n")

    logger.info("Clip timestamps written to %s and %s (%d clips)", csv_path.name, txt_path.name, len(rows))


def main(argv=None) -> int:
    args = parse_args(argv)
    setup_logging(args.log_level)
    cfg = build_config(args)
    cfg.ensure_dirs()

    if not cfg.input_dir.exists():
        logger.error("Input directory does not exist: %s", cfg.input_dir)
        return 2

    # verify STT server reachable before doing any work
    ok, msg = custom_stt.check_server(cfg.custom_stt_url)
    if ok:
        logger.info("Custom STT server check: %s", msg)
    else:
        logger.error("Custom STT server not reachable: %s", msg)
        logger.error("Start the server or fix --stt-url before running. Aborting.")
        return 2

    files = discover_audio_files(cfg.input_dir)
    if cfg.limit is not None:
        files = files[: cfg.limit]
    if not files:
        logger.warning("No audio files found in %s", cfg.input_dir)
        return 0

    logger.info("Found %d source file(s) to process with custom STT.", len(files))
    if cfg.enable_speaker_analysis:
        from pathlib import Path as _P
        _dbp = _P(cfg.speaker_db_path) if cfg.speaker_db_path else (_P(__file__).resolve().parent / "speaker_db_global.json")
        from pipeline import speaker_analysis as _spk
        _db = _spk.SpeakerDatabase(_dbp, cfg.speaker_dup_reject_similarity, cfg.speaker_dup_review_similarity)
        logger.info("Global speaker database: %s (%d speaker(s) already registered)", _dbp, len(_db.entries))
    n_ok = n_failed = 0
    t0 = time.time()
    for i, f in enumerate(files, 1):
        logger.info("=== [%d/%d] %s ===", i, len(files), f.name)
        try:
            process_file(cfg, f)
            n_ok += 1
        except audio_utils.AudioIntegrityError as e:
            logger.error("[%s] INTEGRITY FAILURE: %s", f.name, e); n_failed += 1
        except Exception as e:  # noqa: BLE001
            logger.error("[%s] UNEXPECTED ERROR: %s", f.name, e); logger.error(traceback.format_exc()); n_failed += 1

    run_dataset_postprocessing(cfg)
    logger.info("=== Run complete in %.1f min: %d OK, %d failed. ===", (time.time() - t0) / 60.0, n_ok, n_failed)
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
