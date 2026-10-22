"""Two-phase pipeline.
Phase 1 (prepare_local): validation, audio/frames, transcription, audio metrics - all LOCAL.
Phase 2 (assess_remote): sends transcript + audio stats + sampled frames to the LLM provider,
only when consent_to_send is True. Temp files are removed by LocalEvidence.close()."""
from __future__ import annotations
import logging
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from . import __version__
from .assessment_service import AssessmentError, AssessmentService
from .audio_analyzer import analyze_audio
from .config import Settings
from .media_processor import MediaError, MediaInfo, extract_audio, sample_frames, validate_media
from .rubric import RUBRIC, THRESHOLDS
from .schemas import (AudioFeatures, FinalAssessment, FrameInfo, RunMetadata, Status, Transcript)
from .transcription import Transcriber, TranscriptionError

log = logging.getLogger(__name__)
STAGES = ["Validating media", "Extracting audio and frames", "Transcribing speech (local)",
          "Measuring audio (local)", "Generating and validating the assessment (external LLM)"]
NO_CONSENT_MSG = ("External processing was not approved. Nothing was sent to the LLM provider. "
                  "Approve sending the transcript, audio measurements and sampled frames to proceed.")


@dataclass
class LocalEvidence:
    info: MediaInfo
    transcript: Transcript
    audio: AudioFeatures | None
    frames: list[FrameInfo]
    warnings: list[str] = field(default_factory=list)
    _tmp: tempfile.TemporaryDirectory | None = None

    def disclosure(self) -> dict:
        return {"transcript_words": self.transcript.word_count, "frame_count": len(self.frames),
                "frame_timestamps_s": [f.timestamp_s for f in self.frames],
                "audio_measurements": self.audio is not None}

    def close(self) -> None:
        if self._tmp is not None:
            self._tmp.cleanup()
            self._tmp = None


def prepare_local(video_path: str, settings: Settings, transcriber: Transcriber,
                  progress: Callable[[int, str], None] = lambda i, s: None) -> LocalEvidence:
    tmp = tempfile.TemporaryDirectory(prefix="humaniti_")
    try:
        warnings: list[str] = []
        progress(0, STAGES[0])
        info = validate_media(video_path, settings)
        progress(1, STAGES[1])
        wav = None
        if info.has_audio:
            wav = extract_audio(video_path, Path(tmp.name) / "audio.wav")
        else:
            warnings.append("Recording has no audio track.")
        frames: list[FrameInfo] = []
        if info.has_video:
            try:
                frames = sample_frames(video_path, Path(tmp.name) / "frames", info.duration_s,
                                       settings.frame_interval_s, settings.max_frames_for_llm)
            except MediaError as e:
                warnings.append(str(e))
            if not frames:
                warnings.append("No video frames could be sampled; visual modality unavailable.")
        else:
            warnings.append("Recording has no video track; visual modality unavailable.")
        progress(2, STAGES[2])
        transcript = transcriber.transcribe(str(wav)) if wav else Transcript()
        progress(3, STAGES[3])
        audio = None
        if wav:
            try:
                audio = analyze_audio(str(wav))
            except Exception:
                log.exception("Audio analysis failed")
                warnings.append("Audio feature extraction failed; audio metrics unavailable.")
        return LocalEvidence(info, transcript, audio, frames, warnings, tmp)
    except BaseException:
        tmp.cleanup()
        raise


def build_metadata(settings: Settings, service: AssessmentService, ev: LocalEvidence,
                   result: FinalAssessment) -> RunMetadata:
    sent: list[str] = []
    if result.model_used is not None:  # set only when the LLM was actually called
        sent = ["transcript text", "audio measurements"] if ev.audio else ["transcript text"]
        if ev.frames:
            sent.append(f"{len(ev.frames)} sampled video frame(s)")
    return RunMetadata(
        app_version=__version__, generated_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        whisper_model=settings.whisper_model, whisper_device=settings.whisper_device,
        whisper_compute_type=settings.whisper_compute_type,
        llm_model=result.model_used, weights={d.key: d.weight for d in RUBRIC},
        thresholds={r.value: lo for lo, r in THRESHOLDS}, transcript_words=ev.transcript.word_count,
        frame_count=len(ev.frames), data_sent_to_external_provider=sent)


def assess_remote(ev: LocalEvidence, settings: Settings, service: AssessmentService,
                  consent_to_send: bool) -> FinalAssessment:
    if not consent_to_send:
        return FinalAssessment(status=Status.FAILED, message=NO_CONSENT_MSG, warnings=ev.warnings)
    try:
        result = service.assess(ev.transcript, ev.audio, ev.frames, ev.info.duration_s)
    except AssessmentError as e:
        return FinalAssessment(status=Status.FAILED, message=str(e), warnings=ev.warnings)
    except Exception:
        log.exception("Unexpected assessment failure")
        return FinalAssessment(status=Status.FAILED, warnings=ev.warnings,
                               message="An unexpected error occurred during assessment.")
    result.warnings = ev.warnings + result.warnings
    result.metadata = build_metadata(settings, service, ev, result)
    return result


def run_pipeline(video_path: str, settings: Settings, transcriber: Transcriber, service: AssessmentService,
                 progress: Callable[[int, str], None] = lambda i, s: None,
                 consent_to_send: bool = False) -> FinalAssessment:
    """One-shot convenience (used by tests/CLI). Refuses up-front without consent."""
    if not consent_to_send:
        return FinalAssessment(status=Status.FAILED, message=NO_CONSENT_MSG)
    ev = None
    try:
        ev = prepare_local(video_path, settings, transcriber, progress)
        progress(4, STAGES[4])
        return assess_remote(ev, settings, service, consent_to_send)
    except (MediaError, TranscriptionError, AssessmentError) as e:
        return FinalAssessment(status=Status.FAILED, message=str(e))
    except Exception:
        log.exception("Unexpected pipeline failure")
        return FinalAssessment(status=Status.FAILED,
                               message="An unexpected error occurred while processing the recording.")
    finally:
        if ev:
            ev.close()
