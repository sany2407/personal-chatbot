"""Startup index bootstrap: rebuild only when the store is unusable."""

from types import SimpleNamespace

import pytest

import app.rag.index as index_module


@pytest.fixture
def settings(monkeypatch):
    fake = SimpleNamespace(gemini_embedding_model="gemini-embedding-2")
    monkeypatch.setattr(index_module, "get_settings", lambda: fake)
    return fake


def test_ready_when_index_matches(monkeypatch, settings):
    monkeypatch.setattr(index_module, "indexed_chunks", lambda: 31)
    monkeypatch.setattr(
        index_module,
        "read_index_meta",
        lambda: {"embedding_model": "gemini-embedding-2", "chunks": 31},
    )
    monkeypatch.setattr(
        index_module,
        "rebuild_index",
        lambda: pytest.fail("should not rebuild a valid index"),
    )

    status = index_module.ensure_index()
    assert status == {
        "status": "ready",
        "chunks": 31,
        "embedding_model": "gemini-embedding-2",
    }


def test_missing_meta_is_treated_as_ready(monkeypatch, settings):
    """An index built by the CLI before this feature existed stays usable."""
    monkeypatch.setattr(index_module, "indexed_chunks", lambda: 12)
    monkeypatch.setattr(index_module, "read_index_meta", lambda: {})
    monkeypatch.setattr(
        index_module, "rebuild_index", lambda: pytest.fail("should not rebuild")
    )

    assert index_module.ensure_index()["chunks"] == 12


def test_empty_index_is_rebuilt(monkeypatch, settings):
    monkeypatch.setattr(index_module, "indexed_chunks", lambda: 0)
    monkeypatch.setattr(index_module, "read_index_meta", lambda: {})
    monkeypatch.setattr(index_module, "rebuild_index", lambda: 31)

    status = index_module.ensure_index()
    assert status["status"] == "ready"
    assert status["chunks"] == 31


def test_changed_embedding_model_forces_rebuild(monkeypatch, settings):
    monkeypatch.setattr(index_module, "indexed_chunks", lambda: 31)
    monkeypatch.setattr(
        index_module,
        "read_index_meta",
        lambda: {"embedding_model": "gemini-embedding-001", "chunks": 31},
    )
    monkeypatch.setattr(index_module, "rebuild_index", lambda: 9)

    status = index_module.ensure_index()
    assert status["chunks"] == 9
    assert status["embedding_model"] == "gemini-embedding-2"


def test_failed_rebuild_keeps_the_api_bootable(monkeypatch, settings):
    monkeypatch.setattr(index_module, "indexed_chunks", lambda: 0)
    monkeypatch.setattr(index_module, "read_index_meta", lambda: {})

    def explode():
        raise RuntimeError("gemini quota exhausted")

    monkeypatch.setattr(index_module, "rebuild_index", explode)

    status = index_module.ensure_index()
    assert status["status"] == "error"
    assert "quota" in status["error"]
    assert index_module.index_status()["status"] == "error"
