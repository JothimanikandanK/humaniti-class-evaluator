"""Speech-to-text. Local faster-whisper by default: recordings of children never leave the machine."""
from __future__ import annotations
import logging
from typing import Protocol

from .config import Settings
from .schemas import Transcript, TranscriptSegment

log = logging.getLogger(__name__)


class TranscriptionError(Exception):
    """User-facing transcription failure."""


class Transcriber(Protocol):
    def transcribe(self, wav_path: str) -> Transcript: ...


class FasterWhisperTranscriber:
    def __init__(self, settings: Settings):
        self.s = settings
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
                self._model = WhisperModel(self.s.whisper_model, device=self.s.whisper_device,
                                           compute_type=self.s.whisper_compute_type)
            except Exception as e:  # download failure, missing weights, bad device...
                log.exception("Whisper load failed")
                raise TranscriptionError(
                    "Speech model could not be loaded (first run needs to download the model weights "
                    "from Hugging Face). Check connectivity/WHISPER_MODEL settings.") from e
        return self._model

    def transcribe(self, wav_path: str) -> Transcript:
        model = self._load()
        try:
            segs, info = model.transcribe(wav_path, vad_filter=True, beam_size=5)
            segments = [TranscriptSegment(start=max(0.0, s.start), end=max(0.0, s.end), text=s.text.strip())
                        for s in segs if s.text.strip()]
        except Exception as e:
            log.exception("Transcription failed")
            raise TranscriptionError("Transcription failed while processing the audio.") from e
        return Transcript(language=getattr(info, "language", None), segments=segments)


def format_for_prompt(t: Transcript, max_chars: int = 60000) -> str:
    """Timestamped transcript lines: [mm:ss] text"""
    lines = [f"[{int(s.start)//60:02d}:{int(s.start)%60:02d}] {s.text}" for s in t.segments]
    text = "\n".join(lines)
    return text if len(text) <= max_chars else text[:max_chars] + "\n[TRUNCATED]"
