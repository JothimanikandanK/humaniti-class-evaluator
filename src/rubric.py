"""Configurable rubric + deterministic scoring. The LLM never picks the final rating."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from .schemas import Rating, DimensionAssessment


@dataclass(frozen=True)
class Dimension:
    key: str
    label: str
    weight: float
    anchors: dict  # score -> behavioural anchor


RUBRIC: tuple[Dimension, ...] = (
    Dimension("clarity", "Teaching clarity", 0.30, {
        1: "Explanations are confusing, contradictory or largely inaudible; concepts not broken down.",
        3: "Mostly understandable; some steps skipped or jargon unexplained for the age group.",
        5: "Concepts explained step by step in age-appropriate language, with checks that the explanation landed."}),
    Dimension("content", "Content quality", 0.25, {
        1: "Material has errors or is irrelevant to the stated topic.",
        3: "Mostly accurate and relevant; few or generic examples.",
        5: "Accurate, well-chosen examples/demonstrations tied to the learning goal."}),
    Dimension("engagement", "Engagement and interaction", 0.20, {
        1: "One-way delivery; no questions or responses to the child.",
        3: "Some questions asked; limited waiting for or building on responses.",
        5: "Frequent purposeful questions, wait time, and building on child responses (where observable)."}),
    Dimension("structure", "Lesson structure", 0.15, {
        1: "No discernible opening, progression or close.",
        3: "Recognisable flow but weak transitions or no recap.",
        5: "Clear objective, logical progression, recap/closure."}),
    Dimension("av_usability", "Audio and visual usability", 0.10, {
        1: "Audio or visuals frequently prevent understanding.",
        3: "Usable with occasional noise, glare or unreadable material.",
        5: "Clear audio; readable, relevant visual material."}),
)

# Provisional thresholds (lower bound inclusive). Documented in README; never tuned dynamically.
THRESHOLDS: tuple[tuple[float, Rating], ...] = (
    (3.75, Rating.EXCELLENT),
    (2.75, Rating.GOOD),
    (1.75, Rating.AVERAGE),
    (1.00, Rating.POOR),
)


def rating_for(score: float) -> Rating:
    if score < 1.0 or score > 5.0:
        raise ValueError("score must be within 1.00-5.00")
    for lower, rating in THRESHOLDS:
        if score >= lower:
            return rating
    return Rating.POOR


def weight_of(key: str) -> float:
    for d in RUBRIC:
        if d.key == key:
            return d.weight
    raise KeyError(key)


def compute_overall(
    dims: list[DimensionAssessment], min_eligible_weight: float = 0.5
) -> tuple[Optional[float], Optional[Rating], float]:
    """Weighted mean over eligible dimensions (score present AND sufficient evidence),
    re-normalised by eligible weight. Returns (None, None, w) if eligible weight < minimum."""
    known = {d.key for d in RUBRIC}
    total_w = 0.0
    acc = 0.0
    seen: set[str] = set()
    for d in dims:
        if d.dimension not in known or d.dimension in seen:
            continue
        seen.add(d.dimension)
        if not d.sufficient_evidence or d.score is None:
            continue
        w = weight_of(d.dimension)
        total_w += w
        acc += w * d.score
    if total_w < min_eligible_weight or total_w == 0:
        return None, None, round(total_w, 4)
    raw = acc / total_w
    # Rate on the unrounded value so display rounding can never move a score across a threshold.
    return round(raw, 2), rating_for(min(max(raw, 1.0), 5.0)), round(total_w, 4)


def rubric_prompt_text() -> str:
    lines = []
    for d in RUBRIC:
        lines.append(f'- key "{d.key}" ({d.label}, weight {int(d.weight*100)}%)')
        for s in sorted(d.anchors):
            lines.append(f"    {s}: {d.anchors[s]}")
    return "\n".join(lines)
