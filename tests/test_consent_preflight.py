import json
from pathlib import Path
import pytest
from src.assessment_service import AssessmentService
from src.config import Settings
from src.pipeline import NO_CONSENT_MSG, assess_remote, prepare_local, run_pipeline
from src.preflight import check_api_key, check_ffmpeg, check_whisper_model
from src.report_generator import to_json, to_markdown
from src.schemas import Status, Transcript
from tests.test_assessment_service import QUOTE, TR, FakeLLM, full


class FakeTranscriber:
    calls = 0

    def transcribe(self, p):
        FakeTranscriber.calls += 1
        return TR


def test_no_consent_blocks_everything(synth_video, settings):
    FakeTranscriber.calls = 0
    llm = FakeLLM()
    r = run_pipeline(str(synth_video), settings, FakeTranscriber(), AssessmentService(settings, llm))
    assert r.status == Status.FAILED and r.message == NO_CONSENT_MSG
    assert FakeTranscriber.calls == 0 and llm.calls == 0


def test_assess_remote_refuses_without_consent(synth_video, settings):
    ev = prepare_local(str(synth_video), settings, FakeTranscriber())
    try:
        llm = FakeLLM(full(QUOTE))
        r = assess_remote(ev, settings, AssessmentService(settings, llm), consent_to_send=False)
        assert r.status == Status.FAILED and llm.calls == 0
        assert ev.disclosure()["frame_count"] >= 1
    finally:
        ev.close()


def test_with_consent_has_metadata_and_no_secrets(synth_video, settings):
    r = run_pipeline(str(synth_video), settings, FakeTranscriber(),
                     AssessmentService(settings, FakeLLM(full(QUOTE))), consent_to_send=True)
    m = r.metadata
    assert r.status == Status.COMPLETE and m.llm_model == "fake" and "PROVISIONAL" in m.rubric_status
    assert m.thresholds["Excellent"] == 3.75 and "transcript text" in m.data_sent_to_external_provider
    blob = to_json(r) + to_markdown(r)
    assert "test-key" not in blob and "frame_0" not in blob and "provisional" in blob.lower()


def test_insufficient_transcript_sends_nothing(synth_video, settings):
    class Empty:
        def transcribe(self, p): return Transcript()
    r = run_pipeline(str(synth_video), settings, Empty(), AssessmentService(settings, FakeLLM()), consent_to_send=True)
    assert r.status == Status.INSUFFICIENT_EVIDENCE and r.rating is None
    assert "Insufficient Evidence" in to_markdown(r)


def test_temp_dir_removed_on_close(synth_video, settings):
    ev = prepare_local(str(synth_video), settings, FakeTranscriber())
    frame = Path(ev.frames[0].path); assert frame.exists()
    ev.close()
    assert not frame.exists()


def test_preflight_messages(settings, monkeypatch):
    assert check_ffmpeg().ok
    c = check_api_key(Settings(anthropic_api_key="", llm_provider="anthropic"))
    assert not c.ok and "cannot" in c.detail
    ok = check_api_key(Settings(anthropic_api_key="sk-secret-123", llm_provider="anthropic"))
    assert ok.ok and "sk-secret" not in ok.detail

    def boom(*a, **k): raise OSError("offline")
    monkeypatch.setattr("faster_whisper.utils.download_model", boom)
    w = check_whisper_model(settings)
    assert not w.ok and "check_setup.py --download" in w.detail and not w.blocking


def test_gitignore_protects_private_data():
    txt = (Path(__file__).resolve().parents[1] / ".gitignore").read_text()
    for item in (".env", "reports/", "*.mp4", "*.wav", "samples/"):
        assert item in txt
