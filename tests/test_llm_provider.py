import pytest
from pydantic import ValidationError

from app.agent.graph import extract_text
from app.core.config import Settings


def test_extract_text_from_plain_string():
    assert extract_text("A grounded answer.") == "A grounded answer."


def test_extract_text_from_gemini_content_blocks():
    content = [
        {"type": "thinking", "thinking": "internal reasoning"},
        {"type": "text", "text": "A grounded answer [1]."},
    ]
    assert extract_text(content) == "A grounded answer [1]."


def test_settings_reject_unsupported_provider():
    with pytest.raises(ValidationError):
        Settings(llm_provider="ollama")


def test_gemini_api_key_resolution():
    settings = Settings(llm_provider="gemini", gemini_api_key="", llm_api_key="llm-key")
    assert settings.resolve_gemini_api_key() == "llm-key"

    settings = Settings(
        llm_provider="openai_compatible",
        llm_api_key="groq-key",
        gemini_api_key="gemini-key",
    )
    assert settings.resolve_gemini_api_key() == "gemini-key"

    settings = Settings(
        llm_provider="openai_compatible",
        llm_api_key="groq-key",
        gemini_api_key="",
    )
    assert settings.resolve_gemini_api_key() == ""
