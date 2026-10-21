from functools import lru_cache

from langchain_core.embeddings import Embeddings

from app.core.config import get_settings


@lru_cache
def get_embeddings() -> Embeddings:
    """Return the hosted Gemini embedding model.

    Only Gemini embeddings are supported: no local transformer models are
    downloaded, loaded, or executed.
    """
    settings = get_settings()

    if settings.embedding_provider != "gemini":
        raise RuntimeError(
            f"Unsupported embedding provider: {settings.embedding_provider!r}. "
            "Only 'gemini' is supported."
        )

    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    api_key = settings.resolve_gemini_api_key()
    if not api_key:
        raise RuntimeError(
            "A Gemini API key is required for embeddings. "
            "Set GEMINI_API_KEY (or GOOGLE_API_KEY)."
        )
    return GoogleGenerativeAIEmbeddings(
        model=settings.gemini_embedding_model,
        google_api_key=api_key,
    )
