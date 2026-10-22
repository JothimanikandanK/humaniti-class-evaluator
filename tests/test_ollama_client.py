import json
from src.assessment_service import OllamaLLM, make_llm, AssessmentError
from src.config import Settings

def test_ollama_provider_selected_by_default():
    s = Settings()
    assert s.llm_provider == "ollama"
    assert s.llm_model == "qwen2.5vl:7b"
    assert isinstance(make_llm(s), OllamaLLM)

def test_ollama_payload_supports_text_and_images(monkeypatch):
    captured = {}
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self): return json.dumps({"message": {"content": "{\"dimensions\": []}"}}).encode()
    def fake_urlopen(req, timeout):
        captured["payload"] = json.loads(req.data.decode())
        captured["url"] = req.full_url
        return Response()
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = OllamaLLM(Settings())
    out = client.complete("system prompt", [
        {"type": "text", "text": "evaluate this"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "ZmFrZQ=="}},
    ])
    assert "dimensions" in out
    assert captured["url"].endswith("/api/chat")
    assert captured["payload"]["messages"][0]["role"] == "system"
    assert captured["payload"]["messages"][1]["content"] == "evaluate this"
    assert captured["payload"]["messages"][1]["images"] == ["ZmFrZQ=="]
    assert captured["payload"]["format"] == "json"

def test_anthropic_provider_still_requires_key():
    try:
        make_llm(Settings(llm_provider="anthropic", anthropic_api_key=""))
    except AssessmentError as exc:
        assert "ANTHROPIC_API_KEY" in str(exc)
    else:
        raise AssertionError("Expected missing-key error")
