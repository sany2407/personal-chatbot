import os
import shutil
import stat
import tempfile
from functools import lru_cache
from pathlib import Path

from langchain_chroma import Chroma

from app.core.config import get_settings
from app.rag.embeddings import get_embeddings


def _writable(path: Path) -> bool:
    """True when this process can write the store's SQLite file."""
    try:
        database = path / "chroma.sqlite3"
        if database.exists():
            return os.access(database, os.W_OK)
        return os.access(path, os.W_OK)
    except OSError:
        return False


@lru_cache
def persist_dir() -> Path:
    """Return a Chroma directory this process is allowed to write.

    Hosted builds frequently leave the prebuilt index owned by the build user,
    while the app runs as someone else. Chroma then fails with
    ``attempt to write a readonly database`` even though the files are readable,
    so the index looks empty at boot.

    In that case the prebuilt index is copied to a writable temp directory and
    used from there. Local development keeps using the configured path.
    """
    settings = get_settings()
    configured = Path(settings.chroma_persist_dir)

    try:
        configured.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass

    if _writable(configured):
        return configured

    fallback = Path(tempfile.gettempdir()) / "portfolio-chroma"
    seeded = fallback / "chroma.sqlite3"
    if not seeded.exists() and configured.exists():
        try:
            shutil.copytree(configured, fallback, dirs_exist_ok=True)
            # Copies can inherit read-only bits (notably on Windows); a copied
            # index is useless to Chroma unless this process can write it.
            for item in [fallback, *fallback.rglob("*")]:
                try:
                    os.chmod(item, item.stat().st_mode | stat.S_IWUSR)
                except OSError:
                    pass
            print(f"[chroma] copied read-only index to {fallback}", flush=True)
        except OSError as exc:
            print(f"[chroma] could not copy index to {fallback}: {exc}", flush=True)

    try:
        fallback.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(
            f"No writable Chroma directory: {configured} and {fallback} both failed ({exc})"
        ) from exc
    return fallback


@lru_cache
def get_vectorstore() -> Chroma:
    return Chroma(
        collection_name=get_settings().chroma_collection,
        persist_directory=str(persist_dir()),
        embedding_function=get_embeddings(),
    )
