import pytest
from dataclasses import replace
from src.media_processor import MediaError, extract_audio, sample_frames, validate_media
from src.audio_analyzer import analyze_audio


def test_empty_file(tmp_path, settings):
    p = tmp_path / "a.mp4"; p.write_bytes(b"")
    with pytest.raises(MediaError, match="empty"):
        validate_media(p, settings)


def test_unsupported_extension(tmp_path, settings):
    p = tmp_path / "a.txt"; p.write_text("hi")
    with pytest.raises(MediaError, match="Unsupported"):
        validate_media(p, settings)


def test_corrupt_file(tmp_path, settings):
    p = tmp_path / "a.mp4"; p.write_bytes(b"not a video" * 100)
    with pytest.raises(MediaError, match="corrupt|valid"):
        validate_media(p, settings)


def test_valid_and_limits(synth_video, settings):
    info = validate_media(synth_video, settings)
    assert 5 < info.duration_s < 7 and info.has_audio and info.has_video
    with pytest.raises(MediaError, match="limit"):
        validate_media(synth_video, replace(settings, max_duration_min=0.01))
    with pytest.raises(MediaError, match="limit"):
        validate_media(synth_video, replace(settings, max_upload_mb=0.0001))


def test_extract_audio_frames_and_features(synth_video, tmp_path):
    wav = extract_audio(synth_video, tmp_path / "a.wav")
    f = analyze_audio(str(wav))
    assert 5.5 < f.duration_s < 6.5 and f.speech_ratio > 0.9 and f.clipping_ratio < 0.01
    frames = sample_frames(synth_video, tmp_path / "fr", 6.0, 2.0, 3)
    assert 1 <= len(frames) <= 3 and all(0 <= x.timestamp_s <= 6 for x in frames)
