"""Measurable audio signals from a 16 kHz mono WAV.

Method (documented limitations: energy-based, NOT semantic):
- 30 ms non-overlapping frames; RMS in dBFS = 20*log10(rms/32768), floored at -100.
- noise_floor = 10th percentile of frame dBFS.
- A frame is 'speech-like' if dBFS > max(min(noise_floor + 10 dB, p90 - 6 dB), -50 dBFS).
  (The p90 cap keeps continuous, constant-level audio from being classed as silence.)
- speech_s / silence_s = counts * 30 ms; speech_ratio = speech_s / duration.
- clipping_ratio = share of samples with |x| >= 0.99 full scale.
These say nothing about engagement, emotion or teaching effectiveness.
"""
from __future__ import annotations
import numpy as np
from scipy.io import wavfile

from .schemas import AudioFeatures

FRAME_S = 0.03


def analyze_audio(wav_path: str) -> AudioFeatures:
    sr, data = wavfile.read(wav_path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    x = data.astype(np.float64)
    if x.size < sr * FRAME_S * 2:
        raise ValueError("Audio too short to analyze.")
    full_scale = 32768.0 if data.dtype == np.int16 else max(1.0, float(np.abs(x).max()))
    n = int(sr * FRAME_S)
    frames = x[: (x.size // n) * n].reshape(-1, n)
    rms = np.sqrt((frames ** 2).mean(axis=1))
    db = np.maximum(20 * np.log10(np.maximum(rms, 1e-9) / full_scale), -100.0)
    floor = float(np.percentile(db, 10))
    p90 = float(np.percentile(db, 90))
    speech = db > max(min(floor + 10.0, p90 - 6.0), -50.0)
    duration = x.size / sr
    speech_s = float(speech.sum() * FRAME_S)

    longest = run = 0
    for is_speech in speech:
        run = 0 if is_speech else run + 1
        longest = max(longest, run)

    clip = float((np.abs(x) >= 0.99 * full_scale).mean())
    return AudioFeatures(
        duration_s=round(duration, 1), speech_s=round(speech_s, 1),
        silence_s=round(max(duration - speech_s, 0), 1),
        speech_ratio=round(speech_s / duration, 3), longest_silence_s=round(longest * FRAME_S, 1),
        rms_dbfs_mean=round(float(db.mean()), 1), rms_dbfs_p10=round(floor, 1),
        rms_dbfs_p90=round(p90, 1), noise_floor_dbfs=round(floor, 1),
        clipping_ratio=round(clip, 5))
