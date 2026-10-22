"""Markdown / JSON export. Frame file paths are excluded by the schema."""
from __future__ import annotations
from .schemas import FinalAssessment


def _ts(t: float | None) -> str:
    return "" if t is None else f"[{int(t)//60:02d}:{int(t)%60:02d}] "


def to_json(a: FinalAssessment) -> str:
    return a.model_dump_json(indent=2)


def to_markdown(a: FinalAssessment) -> str:
    L = ["# Class Teaching Quality Assessment", ""]
    if a.status.value == "complete":
        L += [f"**Rating:** {a.rating.value}  |  **Overall score:** {a.overall_score:.2f} / 5.00",
              f"Evidence-weight coverage: {a.eligible_weight:.0%}", ""]
    else:
        label = "Insufficient Evidence" if a.status.value == "insufficient_evidence" else "Failed"
        L += [f"**Result:** {label} (no rating issued)", a.message or "", ""]
    if a.dimensions:
        L += ["## Criteria"]
        for d in a.dimensions:
            sc = f"{d.score}/5" if d.eligible else "insufficient evidence"
            L += [f"### {d.label} ({int(d.weight*100)}%) — {sc}", d.explanation]
            for e in d.evidence:
                q = f' — "{e.quote}"' if e.quote else ""
                L.append(f"- _{e.kind}, {e.source}_ {_ts(e.timestamp_s)}{e.description}{q}")
            L.append("")
    for title, items in (("Strengths", a.strengths), ("Areas for improvement", a.improvements),
                         ("Missing evidence", a.missing_evidence), ("Processing warnings", a.warnings),
                         ("Limitations", a.limitations)):
        if items:
            L += [f"## {title}"] + [f"- {i}" for i in items] + [""]
    if a.evidence_coverage:
        L += ["## Evidence coverage"] + [f"- {k}: {'available' if v else 'unavailable'}"
                                         for k, v in a.evidence_coverage.items()] + [""]
    m = a.metadata
    if m:
        L += ["## Run configuration",
              f"- Generated (UTC): {m.generated_at_utc}  |  App version: {m.app_version}",
              f"- Transcription: faster-whisper `{m.whisper_model}` ({m.whisper_device}, {m.whisper_compute_type}), local",
              f"- LLM: {m.llm_model or 'not called'}",
              f"- Rubric weights: {m.weights}", f"- Rating thresholds (lower bounds): {m.thresholds}",
              f"- **Rubric status: {m.rubric_status}**",
              f"- Transcript words: {m.transcript_words}; frames analysed: {m.frame_count}",
              f"- Sent to external provider: {', '.join(m.data_sent_to_external_provider) or 'nothing'}", ""]
    L.append("_Automated assessment; not scientifically validated against labelled ground truth._")
    return "\n".join(L)
