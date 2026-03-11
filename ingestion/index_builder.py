"""
Build the FAISS vector index from raw documents.

This is the missing production step between:
  data/raw_docs/  ->  ingestion (load, chunk, embed)  ->  data/faiss_index + data/faiss_docstore.pkl
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dotenv import load_dotenv

from ingestion.chunker import chunk_documents
from ingestion.document_loader import load_documents
from ingestion.embedding_pipeline import EmbeddingPipeline
from utils.config_loader import load_config
from utils.logger import get_logger, setup_logger
from vectorstore.faiss_store import FAISSStore


@dataclass(frozen=True)
class BuildStats:
    documents_loaded: int
    chunks_created: int
    vectors_indexed: int


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _resolve_path(p: str | Path) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    return _project_root() / path


def build_faiss_index(
    *,
    config_path: Optional[Path] = None,
    docs_dir: Optional[Path] = None,
) -> Tuple[FAISSStore, BuildStats]:
    """
    Build and persist a FAISS index from documents on disk.

    Args:
        config_path: Optional config override.
        docs_dir: Optional docs directory override (defaults to config paths.raw_docs).

    Returns:
        (store, stats)
    """
    load_dotenv()
    setup_logger()
    logger = get_logger()

    cfg = load_config(config_path)
    raw_docs = docs_dir or _resolve_path(str(cfg["paths"]["raw_docs"]))
    index_path = _resolve_path(str(cfg["vectorstore"]["index_path"]))
    docstore_path = _resolve_path(str(cfg["vectorstore"]["docstore_path"]))

    emb_model = str(cfg["embedding"]["model"])
    dim = int(cfg.get("embedding", {}).get("dim", 384))

    chunk_size = int(cfg["chunking"]["chunk_size"])
    chunk_overlap = int(cfg["chunking"]["chunk_overlap"])
    separators: Sequence[str] = list(cfg["chunking"].get("separators") or [])

    logger.info(f"Building index from: {raw_docs}")
    docs = load_documents(raw_docs)
    if not docs:
        raise ValueError(f"No supported documents found under: {raw_docs}")

    chunks = chunk_documents(
        docs,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators if separators else None,
    )

    texts: List[str] = [c.page_content for c in chunks]
    metadatas: List[Dict[str, Any]] = [dict(c.metadata or {}) for c in chunks]

    cache_path = _resolve_path("data/embedding_cache.sqlite")
    embedder = EmbeddingPipeline(model_name=emb_model, cache_path=cache_path, normalize_embeddings=True)
    vectors = embedder.embed_documents(texts)

    store = FAISSStore(dim=dim, metric="ip", normalize_embeddings=True)
    store.add_texts(texts=texts, embeddings=vectors, metadatas=metadatas)
    store.save(index_path=index_path, docstore_path=docstore_path)

    stats = BuildStats(
        documents_loaded=len(docs),
        chunks_created=len(chunks),
        vectors_indexed=len(texts),
    )
    logger.info(
        "Index build complete "
        f"(docs={stats.documents_loaded}, chunks={stats.chunks_created}, vectors={stats.vectors_indexed})"
    )
    return store, stats


def main() -> None:
    # Minimal CLI entrypoint. Run:
    #   python -m ingestion.index_builder
    build_faiss_index()


if __name__ == "__main__":
    main()

