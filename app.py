"""Streamlit UI. Phase 1 runs locally; Phase 2 (external LLM) requires explicit approval."""
import logging
import os
import tempfile

import streamlit as st

from src.assessment_service import AssessmentService
from src.config import Settings
from src.media_processor import MediaError
from src.pipeline import STAGES, assess_remote, prepare_local
from src.preflight import run_preflight
from src.report_generator import to_json, to_markdown
from src.schemas import Status
from src.transcription import FasterWhisperTranscriber, TranscriptionError

logging.basicConfig(level=logging.INFO)
st.set_page_config(page_title="Class Quality Evaluator", page_icon="🎓", layout="wide")
settings = Settings.from_env()


@st.cache_resource
def get_transcriber():
    return FasterWhisperTranscriber(settings)


def discard():
    ev = st.session_state.pop("evidence", None)
    if ev:
        ev.close()
    st.session_state.pop("result", None)


st.title("🎓 Class Teaching Quality Evaluator")
st.caption("Transcription and audio measurements run locally. Assessment uses your configured LLM provider only after you approve it.")

checks = run_preflight(settings)
with st.sidebar:
    st.subheader("Setup status")
    for c in checks:
        st.markdown(f"{'✅' if c.ok else '❌'} **{c.name}** — {c.detail}")
    st.caption("Rubric weights and thresholds are PROVISIONAL and not scientifically validated.")
ffmpeg_ok, whisper_ok, llm_ok = (c.ok for c in checks)

st.info(f"**Supported:** {', '.join(settings.supported_extensions)} · max {settings.max_upload_mb:.0f} MB · "
        f"max {settings.max_duration_min:.0f} min. Only upload recordings you are authorized to process.")
if not ffmpeg_ok:
    st.error("FFmpeg/ffprobe is missing - install it before analyzing (see sidebar).")
if ffmpeg_ok and not whisper_ok:
    st.warning("The Whisper model is not downloaded yet. The first run will download it (internet needed once).")

up = st.file_uploader("Class recording", type=[e.lstrip(".") for e in settings.supported_extensions],
                      on_change=discard)
if st.button("Step 1 · Process locally", type="primary", disabled=up is None or not ffmpeg_ok):
    discard()
    suffix = os.path.splitext(up.name)[1].lower()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        for chunk in iter(lambda: up.read(8 * 1024 * 1024), b""):
            tmp.write(chunk)
        path = tmp.name
    try:
        with st.status("Processing locally…", expanded=True) as status:
            ev = prepare_local(path, settings, get_transcriber(),
                               lambda i, label: st.write(f"{i+1}/{len(STAGES)} — {label}…"))
            status.update(label="Local processing complete", state="complete")
        st.session_state["evidence"] = ev
    except (MediaError, TranscriptionError) as e:
        st.error(str(e))
    except Exception:
        logging.exception("local processing failed")
        st.error("An unexpected error occurred while processing the recording.")
    finally:
        try:
            os.remove(path)  # original upload no longer needed
        except OSError:
            pass

ev = st.session_state.get("evidence")
if ev and "result" not in st.session_state:
    d = ev.disclosure()
    st.subheader("Local results")
    st.write(f"Transcript: **{d['transcript_words']} words** · language: {ev.transcript.language} · "
             f"duration {ev.info.duration_s/60:.1f} min · {d['frame_count']} frame(s) sampled")
    with st.expander("Preview transcript (stays on this machine until you approve)"):
        st.text(ev.transcript.text[:3000] or "(empty)")
    for w in ev.warnings:
        st.warning(w)
    provider_label = "local Ollama" if settings.llm_provider == "ollama" else "Anthropic"
    st.subheader("Step 2 · Approve LLM assessment")
    st.warning(f"To assess the lesson, the following will be sent to **{provider_label} ({settings.llm_model})**: "
               f"the full transcript text ({d['transcript_words']} words), audio measurements, and "
               f"{d['frame_count']} resized video frame(s) (which may show children). The original video is not sent.")
    ok = st.checkbox("I am authorized to process this recording and I approve sending the data listed above.")
    if not llm_ok:
        st.error(checks[-1].detail)
    c1, c2 = st.columns(2)
    if c1.button("Assess with LLM", type="primary", disabled=not (ok and llm_ok)):
        with st.spinner("Assessing with the LLM…"):
            res = assess_remote(ev, settings, AssessmentService(settings), consent_to_send=ok)
        ev.close()  # frames/audio removed
        st.session_state["result"] = res
        st.rerun()
    if c2.button("Discard recording data"):
        discard()
        st.rerun()

r = st.session_state.get("result")
if r:
    if r.status == Status.FAILED:
        st.error(r.message)
    elif r.status == Status.INSUFFICIENT_EVIDENCE:
        st.warning(f"**Insufficient Evidence — no rating issued.** {r.message}")
    if r.rating:
        c1, c2, c3 = st.columns(3)
        c1.metric("Overall rating (provisional rubric)", r.rating.value)
        c2.metric("Overall score", f"{r.overall_score:.2f} / 5")
        c3.metric("Rubric weight with verified evidence", f"{r.eligible_weight:.0%}")
    if r.dimensions:
        st.subheader("Criteria")
        for col, d in zip(st.columns(len(r.dimensions)), r.dimensions):
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
    if r.metadata:
        with st.expander("Run configuration"):
            st.json(r.metadata.model_dump())
    d1, d2 = st.columns(2)
    d1.download_button("Download JSON", to_json(r), "assessment.json", "application/json")
    d2.download_button("Download Markdown", to_markdown(r), "assessment.md", "text/markdown")
    st.caption("Automated assessment; weights/thresholds are provisional and not validated against ground truth.")
