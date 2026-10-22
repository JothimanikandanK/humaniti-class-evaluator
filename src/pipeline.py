"""Orchestrates the real pipeline with progress callbacks; cleans temp files."""
from __future__ import annotations
import logging
import tempfile
from pathlib import Path
from typing import Callable

from .assessment_service import AssessmentError, AssessmentService
from .audio_analyzer import analyze_audio
from .config import Settings
from .media_processor import MediaError, extract_audio, sample_frames, validate_media
from .schemas import FinalAssessment, Status
from .transcription import Transcriber, TranscriptionError

log = logging.getLogger(__name__)
STAGES = ["Validating media", "Extracting audio and frames", "Transcribing speech",
          "Analyzing evidence", "Generating and validating the assessment"]


def run_pipeline(video_path: str, settings: Settings, transcriber: Transcriber,
                 service: AssessmentService,
                 progress: Callable[[int, str], None] = lambda i, s: None) -> FinalAssessment:
    warnings: list[str] = []
    try:
        with tempfile.TemporaryDirectory(prefix="humaniti_") as tmp:
            progress(0, STAGES[0])
            info = validate_media(video_path, settings)

            progress(1, STAGES[1])
            wav = None
            if info.has_audio:
                wav = extract_audio(video_path, Path(tmp) / "audio.wav")
            else:
                warnings.append("Recording has no audio track.")
            frames = []
            if info.has_video:
                try:
                    frames = sample_frames(video_path, Path(tmp) / "frames", info.duration_s,
                                           settings.frame_interval_s, settings.max_frames_for_llm)
                except MediaError as e:
                    warnings.append(str(e))
                if not frames:
                    warnings.append("No video frames could be sampled; visual modality unavailable.")
            else:
                warnings.append("Recording has no video track; visual modality unavailable.")

            from .schemas import Transcript
            transcript = Transcript()
            progress(2, STAGES[2])
            if wav:
                transcript = transcriber.transcribe(str(wav))

            progress(3, STAGES[3])
            audio = None
            if wav:
                try:
                    audio = analyze_audio(str(wav))
                except Exception:
                    log.exception("Audio analysis failed")
                    warnings.append("Audio feature extraction failed; audio metrics unavailable.")

            progress(4, STAGES[4])
            result = service.assess(transcript, audio, frames, info.duration_s)
            result.warnings = warnings + result.warnings
            return result
    except (MediaError, TranscriptionError, AssessmentError) as e:
        return FinalAssessment(status=Status.FAILED, message=str(e), warnings=warnings)
    except Exception:
        log.exception("Unexpected pipeline failure")
        return FinalAssessment(status=Status.FAILED, warnings=warnings,
                               message="An unexpected error occurred while processing the recording.")
