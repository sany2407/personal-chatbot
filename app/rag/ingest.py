import argparse
import hashlib
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import get_settings
from app.rag.vectorstore import get_vectorstore


SUPPORTED_EXTENSIONS = {".md", ".txt"}


def load_documents(directory: Path) -> list[Document]:
    docs: list[Document] = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        docs.append(Document(
            page_content=text,
            metadata={
                "source": str(path.as_posix()),
                "title": path.stem.replace("-", " ").replace("_", " ").title(),
            },
        ))
    return docs


def ingest() -> int:
    settings = get_settings()
    knowledge_dir = Path(settings.knowledge_dir)
    if not knowledge_dir.exists():
        raise FileNotFoundError(f"Knowledge directory not found: {knowledge_dir}")

    docs = load_documents(knowledge_dir)
    if not docs:
        raise RuntimeError(f"No .md or .txt documents found in {knowledge_dir}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        add_start_index=True,
    )
    chunks = splitter.split_documents(docs)

    # Stable IDs make re-running ingestion idempotent for unchanged content.
    ids = []
    for chunk in chunks:
        source = chunk.metadata.get("source", "")
        start = chunk.metadata.get("start_index", 0)
        digest = hashlib.sha256(
            f"{source}:{start}:{chunk.page_content}".encode("utf-8")
        ).hexdigest()
        ids.append(digest)

    store = get_vectorstore()
    store.add_documents(chunks, ids=ids)
    print(f"Ingested {len(docs)} documents into {len(chunks)} chunks.")
    print("Note: this starter does not remove obsolete chunk IDs when source files change.")
    return len(chunks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index portfolio knowledge into ChromaDB.")
    parser.parse_args()
    chunks = ingest()

    # Record the embedding model so a mismatch is detected on the next boot.
    from app.rag.index import write_index_meta

    write_index_meta(chunks)
