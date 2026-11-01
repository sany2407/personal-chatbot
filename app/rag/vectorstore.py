import os
import shutil
import stat
import tempfile
from functools import lru_cache
from pathlib import Path

from langchain_chroma import Chroma

from app.core.config import get_settings
from app.rag.embeddings import get_embeddings

# Set when the configured directory turned out to be unusable at runtime.
_override: Path | None = None


def _writable(path: Path) -> bool:
    """Probe the exact directory and SQLite file Chroma will write.

    ``os.access`` only inspects permission bits, which is not enough on
    container filesystems: a build artifact can look writable by mode while the
    mount refuses writes. Actually attempting a write is the only reliable test.
    """
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write-probe"
        probe.write_text("", encoding="utf-8")
        probe.unlink()
        database = path / "chroma.sqlite3"
        if database.exists():
            with open(database, "ab"):
                pass
        return True
    except OSError:
        return False


def _seed_copy(source: Path, target: Path) -> None:
    """Copy a prebuilt index, clearing any inherited read-only bits."""
    if not source.exists():
        return
    try:
        shutil.copytree(source, target, dirs_exist_ok=True)
    except OSError as exc:
        print(f"[chroma] could not copy index to {target}: {exc}", flush=True)
        return
    for item in [target, *target.rglob("*")]:
        try:
            os.chmod(item, item.stat().st_mode | stat.S_IWUSR)
        except OSError:
            pass
    print(f"[chroma] copied index to {target}", flush=True)


def temp_store_dir() -> Path:
    """Writable location used when the configured store cannot be written."""
    return Path(tempfile.gettempdir()) / "portfolio-chroma"


def switch_to_temp_store() -> Path:
    """Move the store to a writable temp directory.

    Called after Chroma reports a readonly database: the configured directory
    exists and is readable, but this process cannot write it (typical when a
    build step and the runtime use different users or a read-only image layer).
    The prebuilt index is copied across so no re-embedding is needed.
    """
    global _override

    if _override is not None:
        return _override

    configured = Path(get_settings().chroma_persist_dir)
    fallback = temp_store_dir()
    if configured.resolve() != fallback.resolve():
        if not (fallback / "chroma.sqlite3").exists():
            _seed_copy(configured, fallback)

    fallback.mkdir(parents=True, exist_ok=True)
    _override = fallback
    persist_dir.cache_clear()
    get_vectorstore.cache_clear()
    print(f"[chroma] using writable store at {fallback}", flush=True)
    return fallback


@lru_cache
def persist_dir() -> Path:
    """Return a Chroma directory this process can actually write."""
    if _override is not None:
        return _override

    configured = Path(get_settings().chroma_persist_dir)
    if _writable(configured):
        return configured

    fallback = temp_store_dir()
    _seed_copy(configured, fallback)
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


@lru_cache
def get_vectorstore() -> Chroma:
    return Chroma(
        collection_name=get_settings().chroma_collection,
        persist_directory=str(persist_dir()),
        embedding_function=get_embeddings(),
    )


def store_paths() -> dict:
    """Where the store resolved to, for /health diagnostics."""
    settings = get_settings()
    configured = Path(settings.chroma_persist_dir)
    resolved = persist_dir()
    return {
        "configured": str(configured),
        "configured_writable": _writable(configured),
        "resolved": str(resolved),
        "resolved_writable": _writable(resolved),
        "temp": str(temp_store_dir()),
        "uid": getattr(os, "getuid", lambda: None)(),
    }
