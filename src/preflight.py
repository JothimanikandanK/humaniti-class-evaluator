"""Readiness checks. Never contain or echo secrets; never fall back to mock services."""
from __future__ import annotations
import os
import shutil
import json
import urllib.error
import urllib.request
from dataclasses import dataclass

from .config import Settings


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    blocking: bool = True


def check_ffmpeg() -> Check:
    ok = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
    return Check("FFmpeg + ffprobe", ok, "found on PATH" if ok else
                 "not found. Windows: `winget install Gyan.FFmpeg`, then open a NEW terminal.")


def check_llm_provider(settings: Settings) -> Check:
    provider = settings.llm_provider.strip().lower()
    if provider == "anthropic":
        ok = bool(settings.anthropic_api_key.strip())
        return Check("Anthropic API key", ok, "configured (value hidden)" if ok else
                     "ANTHROPIC_API_KEY is not set. Transcription can run, but a real LLM assessment cannot.")
    if provider != "ollama":
        return Check("LLM provider", False, "Unsupported LLM_PROVIDER. Use 'ollama' or 'anthropic'.")
    try:
        req = urllib.request.Request(f"{settings.ollama_base_url.rstrip('/')}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=2) as response:
            data = json.loads(response.read().decode("utf-8"))
        models = [m.get("name", "") for m in data.get("models", [])]
        found = any(name == settings.llm_model or name.startswith(settings.llm_model + ":") for name in models)
        if found:
            return Check("Ollama", True, f"running; model '{settings.llm_model}' available")
        return Check("Ollama", False, f"running, but model '{settings.llm_model}' is missing. Run `ollama pull {settings.llm_model}`.")
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return Check("Ollama", False, f"not reachable at {settings.ollama_base_url}. Start Ollama and run `ollama pull {settings.llm_model}`.")


def check_api_key(settings: Settings) -> Check:
    """Backward-compatible name used by older tests/callers; checks selected provider."""
    return check_llm_provider(settings)


def check_whisper_model(settings: Settings, download: bool = False) -> Check:
    m = settings.whisper_model
    if os.path.isdir(m):
        return Check("Whisper model", True, f"local folder '{m}'", blocking=False)
    try:
        from faster_whisper.utils import download_model
        download_model(m, local_files_only=not download)
        return Check("Whisper model", True, f"'{m}' available locally", blocking=False)
    except Exception:
        hint = ("download failed - check internet access / WHISPER_MODEL" if download else
                "not downloaded yet. Run `python scripts/check_setup.py --download` (needs internet once).")
        return Check("Whisper model", False, f"'{m}' {hint}", blocking=False)


def run_preflight(settings: Settings, download: bool = False) -> list[Check]:
    return [check_ffmpeg(), check_whisper_model(settings, download), check_api_key(settings)]
