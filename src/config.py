"""Configuration loaded from environment variables (secrets never hard-coded)."""
from __future__ import annotations
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


def _f(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return float(default)


@dataclass(frozen=True)
class Settings:
    anthropic_api_key: str = ""
    llm_model: str = "claude-sonnet-4-6"
    whisper_model: str = "small"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    max_upload_mb: float = 500
    max_duration_min: float = 90
    frame_interval_s: float = 60
    max_frames_for_llm: int = 8
    min_transcript_words: int = 40
    min_eligible_weight: float = 0.5
    supported_extensions: tuple = (".mp4", ".mov", ".m4v", ".webm", ".mkv")

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            llm_model=os.getenv("LLM_MODEL", "claude-sonnet-4-6"),
            whisper_model=os.getenv("WHISPER_MODEL", "small"),
            whisper_device=os.getenv("WHISPER_DEVICE", "cpu"),
            whisper_compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
            max_upload_mb=_f("MAX_UPLOAD_MB", 500),
            max_duration_min=_f("MAX_DURATION_MIN", 90),
            frame_interval_s=_f("FRAME_INTERVAL_S", 60),
            max_frames_for_llm=int(_f("MAX_FRAMES_FOR_LLM", 8)),
            min_transcript_words=int(_f("MIN_TRANSCRIPT_WORDS", 40)),
            min_eligible_weight=_f("MIN_ELIGIBLE_WEIGHT", 0.5),
        )
