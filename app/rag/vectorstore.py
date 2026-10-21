from functools import lru_cache
from pathlib import Path

from langchain_chroma import Chroma

from app.core.config import get_settings
from app.rag.embeddings import get_embeddings


@lru_cache
def get_vectorstore() -> Chroma:
    settings = get_settings()
    Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=settings.chroma_collection,
        persist_directory=settings.chroma_persist_dir,
        embedding_function=get_embeddings(),
    )
