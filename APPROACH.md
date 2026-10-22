**Approach.** Uploaded video is validated with ffprobe, audio extracted to 16 kHz mono, and frames sampled by seeking
(bounded memory). Speech is transcribed locally with faster-whisper (timestamps preserved), so child recordings are not
sent to an STT vendor. Energy-based audio metrics (speech ratio, silence, RMS, clipping) are computed in numpy. Claude
receives the timestamped transcript, metrics and up to eight frames and returns schema-validated JSON scoring five
weighted dimensions (clarity, content, engagement, structure, AV usability). Python then verifies every evidence item
(quotes must appear verbatim in the transcript; visual evidence must match an analysed frame), removes unsupported
items, marks under-evidenced dimensions ineligible, and computes the weighted score and rating deterministically with
documented thresholds. Too little evidence yields "insufficient evidence", not a guess.

**What worked.** 41 automated tests pass (schemas, boundaries, weighting, fabricated-evidence rejection, malformed LLM
output, injection, corrupt media, transcription failure, reports). Real FFmpeg validation, audio extraction, frame
sampling and audio analysis were verified on synthetic clips; the Streamlit app boots.

**What failed / not done.** No sample recordings, API key or Hugging Face access existed in my sandbox, so I could not
run a real end-to-end transcription or LLM assessment, and nothing was deployed. Weights and thresholds are unvalidated
design choices; I did not compare quality levels in the dataset.

**Limitations.** Engagement is only judged from observable speech; silence-based or visual proxies are not treated as
engagement. LLM scores can vary run to run.
