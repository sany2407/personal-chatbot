from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = "development"
    cors_origins_raw: str = "http://localhost:3000,http://127.0.0.1:3000"

    # Chat model provider: "gemini" (Google AI / Gemini API) or
    # "openai_compatible" (Groq, OpenAI, OpenRouter, Together, ...).
    llm_provider: Literal["gemini", "openai_compatible"] = "gemini"
    llm_model: str = "gemini-3.8-flash"
    llm_api_key: str = ""
    llm_base_url: str = "https://api.openai.com/v1"
    llm_temperature: float = 0.2

    # Dedicated Gemini credentials. GEMINI_API_KEY and GOOGLE_API_KEY are both
    # accepted so the same value can be exported under either common name.
    gemini_api_key: str = ""
    google_api_key: str = ""

    # Embeddings use the hosted Gemini embedding model only.
    # No local transformer/onnx models are downloaded or run.
    embedding_provider: Literal["gemini"] = "gemini"
    gemini_embedding_model: str = "gemini-embedding-001"

    chroma_persist_dir: str = "./.chroma"
    chroma_collection: str = "portfolio_knowledge"
    knowledge_dir: str = "./knowledge"
    chunk_size: int = 900
    chunk_overlap: int = 150
    retrieval_k: int = 5
    max_retrieval_attempts: int = 2
    min_context_chars: int = 80

    # Conversation memory: LangGraph checkpoints (question/answer pairs) per
    # session id, persisted in a SQLite file.
    checkpoint_db_path: str = "./.memory/checkpoints.sqlite"
    max_history_tokens: int = 2000

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins_raw.split(",") if origin.strip()]

    def resolve_gemini_api_key(self) -> str:
        """Return the Gemini API key from the most explicit source available."""
        explicit = self.gemini_api_key.strip() or self.google_api_key.strip()
        if explicit:
            return explicit
        # LLM_API_KEY is only a Gemini key when Gemini is the chat provider.
        if self.llm_provider == "gemini" and self.llm_api_key.strip():
            return self.llm_api_key.strip()
        return ""


@lru_cache
def get_settings() -> Settings:
    return Settings()
