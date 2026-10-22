"""Pydantic schemas for evidence, LLM output and the final assessment."""
from __future__ import annotations
from enum import Enum
from typing import Literal, Optional
from pydantic import BaseModel, Field


class Rating(str, Enum):
    POOR = "Poor"
    AVERAGE = "Average"
    GOOD = "Good"
    EXCELLENT = "Excellent"


class Status(str, Enum):
    COMPLETE = "complete"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    FAILED = "failed"


# ---------- extracted evidence ----------
class TranscriptSegment(BaseModel):
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    text: str


class Transcript(BaseModel):
    language: Optional[str] = None
    segments: list[TranscriptSegment] = []

    @property
    def text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments).strip()

    @property
    def word_count(self) -> int:
        return len(self.text.split())


class AudioFeatures(BaseModel):
    duration_s: float
    speech_s: float
    silence_s: float
    speech_ratio: float
    longest_silence_s: float
    rms_dbfs_mean: float
    rms_dbfs_p10: float
    rms_dbfs_p90: float
    noise_floor_dbfs: float
    clipping_ratio: float


class FrameInfo(BaseModel):
    timestamp_s: float
    sharpness: float
    brightness: float
    path: str = Field(default="", exclude=True)  # internal, never exported


# ---------- LLM output ----------
class EvidenceItem(BaseModel):
    kind: Literal["observed", "interpretation"]
    source: Literal["transcript", "audio", "visual"]
    description: str = Field(min_length=3)
    quote: Optional[str] = None          # verbatim transcript quote (verified in code)
    timestamp_s: Optional[float] = Field(default=None, ge=0)


class DimensionAssessment(BaseModel):
    dimension: str
    score: Optional[int] = Field(default=None, ge=1, le=5)
    sufficient_evidence: bool = True
    explanation: str = ""
    evidence: list[EvidenceItem] = []


class LLMAssessment(BaseModel):
    dimensions: list[DimensionAssessment]
    strengths: list[str] = []
    improvements: list[str] = []
    missing_evidence: list[str] = []
    limitations: list[str] = []


# ---------- final result ----------
class DimensionResult(BaseModel):
    dimension: str
    label: str
    weight: float
    score: Optional[int] = None
    eligible: bool
    explanation: str
    evidence: list[EvidenceItem] = []


class RunMetadata(BaseModel):
    app_version: str
    generated_at_utc: str
    whisper_model: str
    whisper_device: str
    whisper_compute_type: str
    llm_model: Optional[str] = None
    rubric_status: str = "PROVISIONAL - weights and thresholds are design choices, not scientifically validated"
    weights: dict[str, float] = {}
    thresholds: dict[str, float] = {}
    transcript_words: int = 0
    frame_count: int = 0
    data_sent_to_external_provider: list[str] = []


class FinalAssessment(BaseModel):
    status: Status
    overall_score: Optional[float] = None
    rating: Optional[Rating] = None
    dimensions: list[DimensionResult] = []
    strengths: list[str] = []
    improvements: list[str] = []
    missing_evidence: list[str] = []
    warnings: list[str] = []
    limitations: list[str] = []
    evidence_coverage: dict[str, bool] = {}
    eligible_weight: float = 0.0
    audio_features: Optional[AudioFeatures] = None
    frames_analyzed: list[FrameInfo] = []
    model_used: Optional[str] = None
    message: Optional[str] = None
    metadata: Optional[RunMetadata] = None
