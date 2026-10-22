"""Streamlit UI for the class teaching-quality evaluator."""
import logging
import os
import tempfile

import streamlit as st

from src.assessment_service import AssessmentService
from src.config import Settings
from src.pipeline import STAGES, run_pipeline
from src.report_generator import to_json, to_markdown
from src.schemas import Status
from src.transcription import FasterWhisperTranscriber

logging.basicConfig(level=logging.INFO)
st.set_page_config(page_title="Class Quality Evaluator", page_icon="🎓", layout="wide")
settings = Settings.from_env()


@st.cache_resource
def get_transcriber():
    return FasterWhisperTranscriber(settings)


st.title("🎓 Class Teaching Quality Evaluator")
st.caption("Upload an online class recording for children. The app transcribes speech, measures audio, "
           "inspects sampled frames, and produces an evidence-grounded rubric assessment.")
st.info(f"**Supported:** {', '.join(settings.supported_extensions)} · max {settings.max_upload_mb:.0f} MB · "
        f"max {settings.max_duration_min:.0f} min.  \n**Privacy:** speech is transcribed locally. Transcript text, "
        "audio statistics and a few sampled frames are sent to the configured LLM provider (Anthropic) for "
        "assessment. Files are processed in a temporary folder and deleted afterwards. Only upload recordings "
        "you are permitted to process.")

up = st.file_uploader("Class recording", type=[e.lstrip(".") for e in settings.supported_extensions])
if st.button("Analyze", type="primary", disabled=up is None):
    suffix = os.path.splitext(up.name)[1].lower()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        for chunk in iter(lambda: up.read(8 * 1024 * 1024), b""):
            tmp.write(chunk)
        path = tmp.name
    try:
        with st.status("Processing…", expanded=True) as status:
            def progress(i, label):
                st.write(f"{i+1}/{len(STAGES)} — {label}…")
            result = run_pipeline(path, settings, get_transcriber(), AssessmentService(settings), progress)
            status.update(label="Done" if result.status != Status.FAILED else "Failed",
                          state="complete" if result.status != Status.FAILED else "error")
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    st.session_state["result"] = result

r = st.session_state.get("result")
if r:
    if r.status == Status.FAILED:
        st.error(r.message)
    elif r.status == Status.INSUFFICIENT_EVIDENCE and r.overall_score is None:
        st.warning(f"**Insufficient evidence — no rating issued.** {r.message}")
    if r.rating:
        c1, c2, c3 = st.columns(3)
        c1.metric("Overall rating", r.rating.value)
        c2.metric("Overall score", f"{r.overall_score:.2f} / 5")
        c3.metric("Rubric weight with verified evidence", f"{r.eligible_weight:.0%}")
    if r.dimensions:
        st.subheader("Criteria")
        cols = st.columns(len(r.dimensions))
        for col, d in zip(cols, r.dimensions):
            col.metric(f"{d.label} ({int(d.weight*100)}%)", f"{d.score}/5" if d.eligible else "n/a")
        for d in r.dimensions:
            with st.expander(f"{d.label} — {d.score if d.eligible else 'insufficient evidence'}"):
                st.write(d.explanation)
                for e in d.evidence:
                    ts = f"`{int(e.timestamp_s)//60:02d}:{int(e.timestamp_s)%60:02d}` " if e.timestamp_s is not None else ""
                    q = f' — “{e.quote}”' if e.quote else ""
                    st.markdown(f"- **{e.kind}** ({e.source}) {ts}{e.description}{q}")
    for title, items in (("Strengths", r.strengths), ("Improvement opportunities", r.improvements),
                         ("Missing evidence", r.missing_evidence), ("Processing warnings", r.warnings),
                         ("Limitations", r.limitations)):
        if items:
            st.subheader(title)
            for i in items:
                st.markdown(f"- {i}")
    if r.evidence_coverage:
        st.subheader("Evidence coverage")
        st.write({k: ("available" if v else "unavailable") for k, v in r.evidence_coverage.items()})
    if r.audio_features:
        with st.expander("Audio measurements"):
            st.json(r.audio_features.model_dump())
    d1, d2 = st.columns(2)
    d1.download_button("Download JSON", to_json(r), "assessment.json", "application/json")
    d2.download_button("Download Markdown", to_markdown(r), "assessment.md", "text/markdown")
    st.caption("Automated assessment; not validated against labelled ground truth.")
