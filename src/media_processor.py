"""Input validation, audio extraction and frame sampling via FFmpeg/ffprobe + OpenCV."""
from __future__ import annotations
import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2

from .config import Settings
from .schemas import FrameInfo

log = logging.getLogger(__name__)


class MediaError(Exception):
    """User-facing media problem (message is safe to display)."""


@dataclass
class MediaInfo:
    duration_s: float
    has_audio: bool
    has_video: bool
    size_mb: float


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def validate_media(path: str | Path, settings: Settings) -> MediaInfo:
    p = Path(path)
    if not ffmpeg_available():
        raise MediaError("FFmpeg/ffprobe is not installed on the server. Install FFmpeg and retry.")
    if not p.exists() or p.stat().st_size == 0:
        raise MediaError("The uploaded file is empty.")
    if p.suffix.lower() not in settings.supported_extensions:
        raise MediaError(
            f"Unsupported file type '{p.suffix}'. Supported: {', '.join(settings.supported_extensions)}.")
    size_mb = p.stat().st_size / 1e6
    if size_mb > settings.max_upload_mb:
        raise MediaError(f"File is {size_mb:.0f} MB; the limit is {settings.max_upload_mb:.0f} MB.")
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(p)],
            capture_output=True, text=True, timeout=60, check=True).stdout
        data = json.loads(out)
    except (subprocess.SubprocessError, json.JSONDecodeError):
        raise MediaError("The file could not be read as a valid media file (it may be corrupt).")
    streams = data.get("streams", [])
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    has_video = any(s.get("codec_type") == "video" for s in streams)
    try:
        duration = float(data.get("format", {}).get("duration", 0) or 0)
    except (TypeError, ValueError):
        duration = 0.0
    if duration <= 0:
        raise MediaError("Could not determine the recording duration; the file may be corrupt.")
    if duration > settings.max_duration_min * 60:
        raise MediaError(
            f"Recording is {duration/60:.0f} min; the limit is {settings.max_duration_min:.0f} min.")
    if not has_audio and not has_video:
        raise MediaError("The file contains no audio or video streams.")
    return MediaInfo(duration, has_audio, has_video, size_mb)


def extract_audio(video: str | Path, out_wav: str | Path) -> Path:
    """16 kHz mono PCM WAV. Starts at t=0 so audio time == original recording time."""
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
           "-c:a", "pcm_s16le", str(out_wav)]
    try:
        subprocess.run(cmd, capture_output=True, timeout=1800, check=True)
    except subprocess.SubprocessError:
        raise MediaError("Audio extraction failed; the recording's audio track may be unreadable.")
    if not Path(out_wav).exists() or Path(out_wav).stat().st_size < 1000:
        raise MediaError("No usable audio could be extracted from the recording.")
    return Path(out_wav)


def sample_frames(video: str | Path, out_dir: str | Path, duration_s: float,
                  interval_s: float, max_frames: int) -> list[FrameInfo]:
    """Seek-based sampling (bounded memory). Timestamps are original-recording seconds."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    interval_s = max(interval_s, 1.0)
    n = max(1, min(max_frames, int(duration_s // interval_s) + 1))
    # evenly spread over the recording, capped at max_frames
    times = [min(duration_s - 0.5, (i + 0.5) * duration_s / n) for i in range(n)]
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise MediaError("Video frames could not be read.")
    frames: list[FrameInfo] = []
    try:
        for t in times:
            cap.set(cv2.CAP_PROP_POS_MSEC, max(t, 0) * 1000)
            ok, img = cap.read()
            if not ok or img is None:
                continue
            h, w = img.shape[:2]
            scale = 768 / max(h, w)
            if scale < 1:
                img = cv2.resize(img, (int(w * scale), int(h * scale)))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            f = out_dir / f"frame_{int(t):06d}.jpg"
            cv2.imwrite(str(f), img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frames.append(FrameInfo(timestamp_s=round(t, 1),
                                    sharpness=round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 1),
                                    brightness=round(float(gray.mean()), 1), path=str(f)))
    finally:
        cap.release()
    return frames
