"""Assess ONE authorized recording. Transcription is local; sending to the LLM needs typed approval.
Usage: python scripts/assess_one.py path\\to\\video.mp4 [--local-only] [--approve-external] [--out-dir reports]"""
import argparse
import hashlib
import sys
from datetime import datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.assessment_service import AssessmentService
from src.config import Settings
from src.media_processor import MediaError
from src.pipeline import assess_remote, prepare_local
from src.preflight import check_api_key, check_ffmpeg
from src.report_generator import to_json, to_markdown
from src.transcription import FasterWhisperTranscriber, TranscriptionError


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--local-only", action="store_true", help="validate + transcribe locally; send nothing")
    ap.add_argument("--approve-external", action="store_true", help="skip the typed prompt (you accept the disclosure)")
    ap.add_argument("--out-dir", default="reports")
    a = ap.parse_args()
    s = Settings.from_env()

    ff = check_ffmpeg()
    if not ff.ok:
        print("ERROR:", ff.detail); return 2
    key = check_api_key(s)
    if not a.local_only and not key.ok:
        print("ERROR:", key.detail, "\nRe-run with --local-only to validate transcription without the LLM."); return 2

    try:
        ev = prepare_local(a.video, s, FasterWhisperTranscriber(s), lambda i, l: print(f"[{i+1}/5] {l}..."))
    except (MediaError, TranscriptionError) as e:
        print("ERROR:", e); return 1
    try:
        d = ev.disclosure()
        print(f"\nLocal result: {d['transcript_words']} words, language={ev.transcript.language}, "
              f"duration={ev.info.duration_s:.0f}s, frames={d['frame_count']}, audio_metrics={d['audio_measurements']}")
        for w in ev.warnings: print("WARNING:", w)
        if a.local_only:
            print("Local-only run complete. Nothing was sent externally."); return 0
        print(f"\nDISCLOSURE: this will send the transcript ({d['transcript_words']} words), audio measurements and "
              f"{d['frame_count']} video frame(s) (may show children) to Anthropic ({s.llm_model}).")
        if not a.approve_external and input("Type SEND to approve (anything else cancels): ").strip() != "SEND":
            print("Cancelled. Nothing was sent."); return 0
        res = assess_remote(ev, s, AssessmentService(s), consent_to_send=True)
    finally:
        ev.close()

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    tag = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + hashlib.sha1(a.video.encode()).hexdigest()[:6]
    (out / f"assessment_{tag}.json").write_text(to_json(res), encoding="utf-8")
    (out / f"assessment_{tag}.md").write_text(to_markdown(res), encoding="utf-8")
    print(f"\nStatus: {res.status.value}  Rating: {res.rating.value if res.rating else 'Insufficient Evidence / n/a'}"
          f"  Score: {res.overall_score}\nReport: {out / ('assessment_' + tag + '.md')}")
    if res.message: print(res.message)
    return 0 if res.status.value != "failed" else 1


if __name__ == "__main__":
    sys.exit(main())
