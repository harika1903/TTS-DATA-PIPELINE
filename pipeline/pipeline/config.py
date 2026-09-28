"""Configuration and constants for the TTS data pipeline."""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Optional

PIPELINE_VERSION = "1.0.0"


@dataclasses.dataclass
class PipelineConfig:
    input_dir: Path
    output_dir: Path
    limit: Optional[int] = None
    force: bool = False

    # --- language / model ---
    language: Optional[str] = "en"          # None = auto-detect
    whisper_model: str = "small"
    whisper_compute_type_gpu: str = "float16"
    whisper_compute_type_cpu: str = "int8"
    device: str = "auto"                    # auto | cpu | cuda
    beam_size: int = 5
    min_language_probability: float = 0.6

    # --- analysis copy ---
    analysis_sr: int = 16000

    # --- segmentation ---
    min_clip_sec: float = 3.0
    max_clip_sec_preferred: float = 12.0
    max_clip_sec_hard: float = 20.0
    max_internal_pause_sec: float = 1.2       # pauses longer than this get trimmed
    trimmed_pause_target_sec: float = 0.4     # trimmed pauses are capped to this, not deleted
    max_silence_between_merge_sec: float = 0.6  # merge adjacent whisper segments if gap <= this

    # --- streaming quality metrics ---
    metrics_block_seconds: float = 1.0
    silence_dbfs_threshold: float = -45.0
    clip_amplitude_threshold: float = 0.998

    # --- per-clip clipping/distortion gate (protocol 5) ---
    # Even brief clipping is audibly distorted and undesirable in TTS
    # training data, so these are intentionally strict: any clip with more
    # than clipping_reject_ratio of its samples at/above
    # clip_amplitude_threshold is rejected outright; anything above
    # clipping_review_ratio but below the reject line is quarantined.
    clipping_reject_ratio: float = 0.001    # 0.1% of samples clipped -> reject
    clipping_review_ratio: float = 0.0001   # 0.01% of samples clipped -> review

    # --- background / noise heuristics (see pipeline/background.py docstring
    #     for the explicit reliability caveat on these) ---
    # NOTE (calibration from real-audio testing): the estimated SNR here is
    # speech-level-minus-noise-floor, where the noise floor comes from the
    # quietest frames. On clean CONTINUOUS speech there is very little true
    # silence to measure a noise floor against, so this estimate reads
    # artificially LOW (e.g. clean Hindi speech measured ~6 dB). Using it as
    # a hard reject wrongly threw out clean audio. It is therefore tuned
    # conservatively and, when DNSMOS is enabled, DNSMOS is the real quality
    # gate and this SNR estimate only informs review. Defaults below reject
    # only genuinely bad SNR and send the merely-questionable range to review.
    snr_reject_db: float = 3.0     # estimated SNR below this -> reject (very low bar; real noise only)
    snr_review_db: float = 8.0     # estimated SNR below this -> human review
    flatness_music_threshold: float = 0.35

    # --- VAD (bundled Silero VAD inside faster-whisper) ---
    vad_min_silence_ms: int = 200
    vad_speech_pad_ms: int = 100

    # --- hallucination thresholds (standard whisper decoder heuristics) ---
    whisper_no_speech_prob_threshold: float = 0.6
    whisper_avg_logprob_threshold: float = -1.0
    whisper_compression_ratio_threshold: float = 2.4

    # --- alignment ---
    min_mean_word_confidence_reject: float = 0.25   # below this -> reject (see alignment.py)
    min_mean_word_confidence_review: float = 0.5    # below this -> review
    min_chars_per_sec: float = 3.0
    max_chars_per_sec: float = 25.0

    # --- loudness ---
    loudness_normalize: bool = True
    target_lufs: float = -23.0
    max_loudness_gain_db: float = 12.0

    # --- dedup ---
    dedup_text_similarity_threshold: float = 0.92
    dedup_audio_similarity_threshold: float = 0.985
    dedup_duration_tolerance_sec: float = 0.75

    # --- speaker verification (optional, off by default - see speaker.py) ---
    enable_speaker_diarization: bool = False
    hf_token: Optional[str] = None
    # If True, a clip whose speaker verification never ran (because
    # enable_speaker_diarization is False) is downgraded to "quarantined"
    # purely for that reason. Defaults to False: with diarization off (the
    # default), this check simply isn't part of the pipeline rather than
    # being treated as "uncertain" - otherwise EVERY clip would be
    # quarantined by default regardless of actual quality, which would make
    # the pipeline produce an empty accepted set out of the box. Set this to
    # True only if you've deliberately decided you want every clip
    # human-reviewed for single-speaker-ness and haven't enabled the real
    # (pyannote-based) check.
    require_speaker_check_for_acceptance: bool = False

    # --- dataset split ---
    split_train_ratio: float = 0.9
    split_val_ratio: float = 0.05
    split_seed: int = 42

    # --- NEW optional tools (all off/absent by default; see check_environment.py) ---
    # Each is only used if its library + model are available; otherwise the
    # pipeline falls back to its built-in behavior. None of these are
    # required for the core pipeline to run.
    enable_dnsmos: bool = False              # objective quality scoring
    dnsmos_model_path: Optional[str] = None  # path to sig_bak_ovr.onnx
    dnsmos_reject_ovrl: float = 2.5          # reject clips below this overall MOS
    dnsmos_review_ovrl: float = 3.0          # review clips below this (Emilia uses ~3.0)

    # The built-in SNR/spectral-flatness background heuristic is unreliable on
    # clean continuous speech (it reads SNR artificially low and mistakes
    # tonal speech for background music -- confirmed by real-audio testing).
    # By default it is ADVISORY ONLY: it records its findings in the clip's
    # metadata but does NOT gate accept/quarantine/reject. Turn it back into a
    # hard gate only if you don't have DNSMOS and want *some* automated noise
    # check. When DNSMOS is enabled, DNSMOS is the real quality gate.
    background_heuristic_gates: bool = False

    enable_source_separation: bool = False   # Demucs, only on music-flagged clips
    enable_number_normalization: bool = False  # NeMo, language-aware
    enable_forced_alignment: bool = False    # torchaudio Wav2Vec2 (rigorous alignment)

    # language: None here means "detect from filename prefix" (hin_1.wav -> Hindi)
    detect_language_from_filename: bool = True

    # --- CUSTOM STT (approach B: team's model + silence cutting) ---
    # When use_custom_stt is True, the pipeline does NOT use Whisper. Instead
    # it cuts clips at silence (no timestamps needed) and sends each clip to
    # the custom STT server for its text.
    use_custom_stt: bool = False
    custom_stt_url: str = "http://0.0.0.0:8123"
    custom_stt_timeout_sec: float = 120.0
    # silence-detection parameters (see silence_segmentation.py)
    silence_db_threshold: float = -38.0
    min_silence_sec: float = 0.35
    # Flag clips containing digits for manual number-normalization review.
    # OFF by default: when off, clean clips that merely contain a number
    # (e.g. "21", "2024") are ACCEPTED instead of quarantined. Turn this back
    # ON if/when you add a number-to-words step (e.g. indic-numtowords) and
    # want those clips held for that conversion. With it off, numbers stay in
    # whatever form the STT model produced (often digits).
    flag_numbers_for_review: bool = False

    # --- SPEAKER ANALYSIS (pyannote: one-speaker-per-file + cross-file dedup) ---
    # Both checks run at the FILE level, BEFORE clip-cutting, so bad files are
    # rejected upfront. Off by default (needs pyannote + a HF token).
    enable_speaker_analysis: bool = False
    hf_token: Optional[str] = None
    speaker_device: str = "cuda"          # pyannote runs on GPU when available
    max_speakers_per_file: int = 1        # files with more speakers than this are rejected
    # cross-file voiceprint similarity (cosine, 0..1). Tune on your data.
    speaker_dup_reject_similarity: float = 0.75   # >= this -> same speaker -> reject file
    speaker_dup_review_similarity: float = 0.60   # >= this (but < reject) -> borderline -> review

    log_level: str = "INFO"

    # ---- derived paths ----
    @property
    def cache_dir(self) -> Path:
        return self.output_dir / ".cache"

    @property
    def state_dir(self) -> Path:
        return self.output_dir / ".state"

    @property
    def clips_dir(self) -> Path:
        return self.output_dir / "clips"

    @property
    def transcripts_dir(self) -> Path:
        return self.output_dir / "transcripts"

    @property
    def quarantine_dir(self) -> Path:
        return self.output_dir / "quarantine"

    @property
    def reports_dir(self) -> Path:
        return self.output_dir / "reports"

    @property
    def manifests_dir(self) -> Path:
        return self.output_dir / "manifests"

    def ensure_dirs(self) -> None:
        for d in [
            self.output_dir, self.cache_dir, self.state_dir, self.clips_dir,
            self.transcripts_dir, self.quarantine_dir, self.reports_dir,
            self.manifests_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)
