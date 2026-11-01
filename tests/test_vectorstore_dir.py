"""The vector store must never open a directory the process cannot write.

Hosted builds leave the prebuilt index owned by the build user, which makes
Chroma fail with "attempt to write a readonly database" at runtime.
"""

from types import SimpleNamespace

import pytest

import app.rag.vectorstore as vectorstore


@pytest.fixture(autouse=True)
def clear_caches():
    yield
    vectorstore.persist_dir.cache_clear()


def test_uses_configured_dir_when_writable(tmp_path, monkeypatch):
    configured = tmp_path / "chroma"
    monkeypatch.setattr(
        vectorstore,
        "get_settings",
        lambda: SimpleNamespace(chroma_persist_dir=str(configured)),
    )
    vectorstore.persist_dir.cache_clear()

    assert vectorstore.persist_dir() == configured
    assert configured.is_dir()


def test_falls_back_to_temp_dir_and_copies_the_seed(tmp_path, monkeypatch):
    seed = tmp_path / "chroma"
    seed.mkdir()
    (seed / "chroma.sqlite3").write_text("index-bytes", encoding="utf-8")
    (seed / "index-meta.json").write_text("{}", encoding="utf-8")
    temp_root = tmp_path / "tmp"
    temp_root.mkdir()

    monkeypatch.setattr(
        vectorstore,
        "get_settings",
        lambda: SimpleNamespace(chroma_persist_dir=str(seed)),
    )
    monkeypatch.setattr(vectorstore, "_writable", lambda path: False)
    monkeypatch.setattr(vectorstore.tempfile, "gettempdir", lambda: str(temp_root))
    vectorstore.persist_dir.cache_clear()

    resolved = vectorstore.persist_dir()

    assert resolved == temp_root / "portfolio-chroma"
    assert (resolved / "chroma.sqlite3").read_text(encoding="utf-8") == "index-bytes"


def test_missing_configured_dir_is_created_when_allowed(tmp_path, monkeypatch):
    configured = tmp_path / "nested" / "chroma"
    monkeypatch.setattr(
        vectorstore,
        "get_settings",
        lambda: SimpleNamespace(chroma_persist_dir=str(configured)),
    )
    vectorstore.persist_dir.cache_clear()

    assert vectorstore.persist_dir() == configured
