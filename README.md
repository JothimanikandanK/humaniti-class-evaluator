# Class Teaching Quality Evaluator (Humaniti AI challenge)

Streamlit app that evaluates an online class recording for children: real transcription, measured audio
features, sampled frames, an evidence-grounded LLM assessment, and **deterministic** rubric scoring.

## Setup (Windows PowerShell)
```powershell
winget install Gyan.FFmpeg                 # then OPEN A NEW TERMINAL
ffmpeg -version; ffprobe -version          # both must print a version

cd humaniti-class-evaluator
py -3.11 -m venv .venv                     # Python 3.10-3.12
.\.venv\Scripts\Activate.ps1               # if blocked: Set-ExecutionPolicy -Scope Process Bypass
pip install -r requirements.txt
winget install --id Ollama.Ollama -e       # if Ollama is not installed
ollama pull qwen2.5vl:7b
Copy-Item .env.example .env                # default uses local Ollama; no API key needed

pytest -q                                  # run the automated test suite
python scripts\check_setup.py --download   # downloads the Whisper model once (needs internet)
streamlit run app.py
```
macOS/Linux: `python3 -m venv .venv && source .venv/bin/activate`, other steps identical (`/` instead of `\`).

## Safe end-to-end procedure (ONE authorized recording first)
1. **Local-only validation** (nothing leaves your machine):
   `python scripts\assess_one.py "C:\path\to\class.mp4" --local-only`
   Confirms media validation, real transcription, word count/language, audio metrics.
2. **Real assessment** - prints exactly what will be sent and requires typing `SEND`:
   `python scripts\assess_one.py "C:\path\to\class.mp4"`
   Reports are written to `reports\` (git-ignored; they can quote the transcript).
3. **UI alternative**: `streamlit run app.py` -> *Step 1 Process locally* -> review -> tick the approval box ->
   *Assess with LLM*.

**Data disclosure.** Transcription and audio measurement are 100 % local. By default, the LLM step uses local Ollama at `OLLAMA_BASE_URL`, so the transcript, audio statistics, and sampled frames are sent to the local Ollama service on your machine, not Anthropic. The original video is never sent. Nothing is sent to the LLM without explicit approval. If you choose `LLM_PROVIDER=anthropic`, the same evidence is sent to Anthropic and an API key is required. If the model, key or media is unavailable the app
shows an error - it never substitutes mock transcription or mock assessments.

## Architecture
`media_processor` (ffprobe validation, 16 kHz audio, seek-based frames) → `transcription` (local faster-whisper,
timestamped) → `audio_analyzer` (energy-based speech/silence, RMS dBFS, clipping) → `assessment_service`
(Ollama text+vision (or Anthropic if configured) → Pydantic validation → **code-level evidence verification** → `rubric.compute_overall`) →
`report_generator` (JSON/Markdown). `pipeline.py` orchestrates with real progress callbacks and temp-dir cleanup.
Visual analysis = local frame metrics (sharpness, brightness) + selected model vision in the same assessment call.

## Rubric (PROVISIONAL — not validated; no labelled data was available)
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
"Speech model could not be loaded" → no access to Hugging Face / wrong `WHISPER_MODEL`. "Ollama not reachable" → start Ollama and check `OLLAMA_BASE_URL`. "Ollama model is missing" → run `ollama pull qwen2.5vl:7b`. For Anthropic, set `LLM_PROVIDER=anthropic` and configure `ANTHROPIC_API_KEY`. "FFmpeg not installed" → install FFmpeg. Upload limit also set in `.streamlit/config.toml`.

## Reports
Each real report contains rubric scores, rating (Poor/Average/Good/Excellent or **Insufficient Evidence**), verified
timestamped evidence, warnings, limitations and a run-configuration block (Whisper + LLM model, weights, thresholds,
what was sent externally). Never commit `reports/`, recordings, transcripts or `.env`.
