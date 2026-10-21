"""Durable conversation memory backed by SQLite.

LangGraph's SQLite checkpointer stores the graph state per thread, so a
conversation survives restarts. The SQLite file is the single source of truth
for chat history, which keeps the API stateless (no in-process session dict)
and works with a single container or worker.
"""

import sqlite3
from functools import lru_cache
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from app.core.config import get_settings


@lru_cache
def get_checkpointer() -> SqliteSaver:
    """Return the process-wide SQLite checkpointer.

    The connection is opened once and kept open for the lifetime of the
    process, so it must not be used as a context manager.
    """
    settings = get_settings()
    path = Path(settings.checkpoint_db_path)
    if path.parent != Path(""):
        path.parent.mkdir(parents=True, exist_ok=True)

    # check_same_thread=False: FastAPI runs sync endpoints in a thread pool.
    conn = sqlite3.connect(str(path), check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    return saver
