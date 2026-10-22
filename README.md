# Class Teaching Quality Evaluator (Humaniti AI challenge)

Streamlit app that evaluates an online class recording for children: real transcription, measured audio
features, sampled frames, an evidence-grounded LLM assessment, and **deterministic** rubric scoring.

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # requires FFmpeg/ffprobe on PATH (apt install ffmpeg)
cp .env.example .env                     # set ANTHROPIC_API_KEY
streamlit run app.py
pytest -q
```
First run downloads Whisper weights (`WHISPER_MODEL`, default `small`) from Hugging Face; later runs are offline.

## Architecture
`media_processor` (ffprobe validation, 16 kHz audio, seek-based frames) → `transcription` (local faster-whisper,
timestamped) → `audio_analyzer` (energy-based speech/silence, RMS dBFS, clipping) → `assessment_service`
(Claude text+vision → Pydantic validation → **code-level evidence verification** → `rubric.compute_overall`) →
`report_generator` (JSON/Markdown). `pipeline.py` orchestrates with real progress callbacks and temp-dir cleanup.
Visual analysis = local frame metrics (sharpness, brightness) + Claude vision in the same assessment call.

## Rubric (provisional — not validated; no labelled data was available)
Clarity 30 %, Content 25 %, Engagement 20 %, Structure 15 %, AV usability 10 %; scores 1–5 with anchors in `rubric.py`.
Overall = weighted mean of *eligible* dimensions (score present and ≥1 verified observed evidence item),
re-normalised. No rating if eligible weight < 50 % (`MIN_ELIGIBLE_WEIGHT`) or transcript < `MIN_TRANSCRIPT_WORDS`.
Thresholds (lower bound inclusive, applied to the unrounded score): Poor 1.00–<1.75, Average <2.75, Good <3.75, Excellent ≥3.75.

## Evidence verification
Transcript "observed" evidence must contain a quote found verbatim (normalised) in the transcript; visual evidence must
cite a timestamp within 2 s of an analysed frame; timestamps beyond the recording are removed. Failures are dropped and
reported as warnings. Transcript text is wrapped as untrusted data in the prompt, and the LLM never outputs the rating.

## Limits / privacy
Audio is transcribed locally. Transcript, audio statistics and ≤8 resized frames go to Anthropic — confirm this is permitted
for your recordings. Do not commit or publicly deploy recordings. Audio metrics never imply engagement or quality.
Hosting needs FFmpeg + enough RAM/CPU for Whisper; Streamlit Community Cloud may be too small for `small` — use `base`/`tiny`.

## Troubleshooting
"Speech model could not be loaded" → no access to Hugging Face / wrong `WHISPER_MODEL`. "ANTHROPIC_API_KEY is not
configured" → set `.env`. "FFmpeg not installed" → install FFmpeg. Upload limit also set in `.streamlit/config.toml`.
