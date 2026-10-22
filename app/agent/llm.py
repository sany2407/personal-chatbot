from functools import lru_cache

from langchain_core.language_models import BaseChatModel

from app.core.config import get_settings

# Gemini 3+ models use fixed sampling defaults and ignore `temperature`.
FIXED_SAMPLING_PREFIXES = ("gemini-3", "gemini-4")


@lru_cache
def get_chat_model() -> BaseChatModel:
    settings = get_settings()

    if settings.llm_provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = settings.resolve_gemini_api_key()
        if not api_key:
            raise RuntimeError(
                "A Gemini API key is required for the gemini provider. "
                "Set GEMINI_API_KEY (or GOOGLE_API_KEY / LLM_API_KEY)."
            )
        kwargs = {
            "model": settings.llm_model,
            "google_api_key": api_key,
            "max_retries": settings.llm_max_retries,
            "timeout": settings.llm_timeout,
        }
        if not settings.llm_model.startswith(FIXED_SAMPLING_PREFIXES):
            kwargs["temperature"] = settings.llm_temperature
        return ChatGoogleGenerativeAI(**kwargs)

    from langchain_openai import ChatOpenAI

    if not settings.llm_api_key.strip():
        raise RuntimeError(
            "LLM_API_KEY is required for the openai_compatible provider. "
            "Use LLM_PROVIDER=gemini with a Gemini API key instead."
        )
    return ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
    )
