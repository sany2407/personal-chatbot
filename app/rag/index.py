"""Build and verify the vector index at startup.

Hosted platforms (Render, Railway, Fly) start each instance from a fresh
filesystem, so the Chroma directory created by ``python -m app.rag.ingest`` is
not guaranteed to exist when the process boots. This module checks the store
and rebuilds it when it is missing, empty, or was built with a different
embedding model.

The check is cheap when the index is already present (one SQLite count), which
keeps cold starts fast.
"""

import json
import shutil
from pathlib import Path

from app.core.config import get_settings
from app.rag.vectorstore import get_vectorstore, persist_dir, switch_to_temp_store

META_FILENAME = "index-meta.json"

# Set by ensure_index() so /health can report the state without touching disk.
_last_status: dict = {}

READONLY_MARKERS = ("readonly database", "read-only", "read only")


def _is_readonly_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(marker in message for marker in READONLY_MARKERS)


def _meta_path() -> Path:
    return persist_dir() / META_FILENAME


def read_index_meta() -> dict:
    """Return the metadata written by the last successful ingestion."""
    try:
        return json.loads(_meta_path().read_text(encoding="utf-8"))
    except (OSError, ValueError, RuntimeError):
        return {}


def write_index_meta(chunks: int) -> None:
    settings = get_settings()
    path = _meta_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "embedding_model": settings.gemini_embedding_model,
                "chunks": chunks,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def indexed_chunks() -> int:
    """Return how many chunks are currently stored, or 0 when unavailable."""
    try:
        return int(get_vectorstore()._collection.count())
    except Exception as exc:
        if _is_readonly_error(exc):
            # The store is readable but this process cannot write it: move to a
            # writable copy and try again before declaring the index empty.
            try:
                switch_to_temp_store()
                return int(get_vectorstore()._collection.count())
            except Exception as retry_exc:
                print(f"[index] writable store retry failed: {retry_exc}", flush=True)
        # A missing or unreadable store is treated as "needs indexing".
        return 0


def rebuild_index() -> int:
    """Drop the persisted store and index the knowledge directory again."""
    # Release the cached Chroma handle before removing its files, otherwise the
    # open SQLite connection keeps the directory locked on Windows.
    get_vectorstore.cache_clear()
    shutil.rmtree(persist_dir(), ignore_errors=True)

    # Imported here so `python -m app.rag.ingest` can import this module
    # without a circular dependency at import time.
    from app.rag.ingest import ingest

    chunks = ingest()
    write_index_meta(chunks)
    return chunks


def _safe_error(exc: Exception) -> str:
    """One-line error text for /health, with anything key-like redacted."""
    import re

    message = f"{type(exc).__name__}: {exc}".replace("\n", " ")
    message = re.sub(r"(key=)[^\s&,)\]]+", r"\1***", message, flags=re.IGNORECASE)
    return message[:300]


def ensure_index(force: bool = False) -> dict:
    """Make sure the vector store matches the configured knowledge.

    Returns a small status dict, also exposed through ``/health``.
    """
    global _last_status

    settings = get_settings()
    model = settings.gemini_embedding_model
    meta = read_index_meta()

    if not force:
        count = indexed_chunks()
        if count > 0:
            if meta.get("embedding_model") in (None, model):
                _last_status = {
                    "status": "ready",
                    "chunks": count,
                    "embedding_model": model,
                }
                return _last_status
            # Vectors from another model cannot be queried with this one.
            print(
                f"[index] embedding model changed "
                f"({meta.get('embedding_model')} -> {model}); rebuilding",
                flush=True,
            )

    print("[index] building vector index from the knowledge directory", flush=True)
    try:
        chunks = rebuild_index()
    except Exception as exc:  # keep the API up; /health reports the failure
        _last_status = {
            "status": "error",
            "chunks": 0,
            "embedding_model": model,
            "error": _safe_error(exc),
        }
        print(f"[index] build failed: {exc}", flush=True)
        return _last_status

    _last_status = {"status": "ready", "chunks": chunks, "embedding_model": model}
    print(f"[index] ready with {chunks} chunks", flush=True)
    return _last_status


def index_status() -> dict:
    """Return the last known index status without touching the store."""
    return dict(_last_status)
