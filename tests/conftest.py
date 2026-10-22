import subprocess
import pytest
from src.config import Settings


@pytest.fixture
def settings():
    return Settings(anthropic_api_key="test-key", min_transcript_words=5)


@pytest.fixture(scope="session")
def synth_video(tmp_path_factory):
    """Real 6 s clip: test pattern + 440 Hz tone (synthetic, no private data)."""
    p = tmp_path_factory.mktemp("media") / "clip.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=10:duration=6",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=6", "-shortest", "-pix_fmt", "yuv420p",
                    str(p)], check=True)
    return p
