"""
RAG pipeline: retrieval + context injection + LLM generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dotenv import load_dotenv

from ingestion.embedding_pipeline import EmbeddingPipeline
from models.llm_model import LLM, get_llm
from utils.config_loader import load_config
from utils.logger import get_logger, setup_logger
from vectorstore.faiss_store import FAISSStore


@dataclass(frozen=True)
class SourceChunk:
    text: str
    metadata: Dict[str, Any]
    score: float


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _resolve_path(p: str | Path) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    return _project_root() / path


def _format_sources(chunks: Sequence[SourceChunk], *, max_chars: int = 1200) -> str:
    parts: List[str] = []
    for i, ch in enumerate(chunks, start=1):
        src = str(ch.metadata.get("source", "unknown"))
        page = ch.metadata.get("page")
        locator = f"{src}" + (f" (page {page})" if page is not None else "")
        snippet = ch.text[:max_chars]
        parts.append(f"[{i}] {locator}\n{snippet}")
    return "\n\n".join(parts)


def _build_prompt(question: str, context: str) -> str:
    return (
        "You are a helpful enterprise assistant. Answer the question using ONLY the context below.\n"
        "If the context is insufficient, say you don't know and ask a clarifying question.\n\n"
        f"## Context\n{context}\n\n"
        f"## Question\n{question}\n\n"
        "## Answer\n"
    )


class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline.

    Loads config, embeds query, retrieves top-k context from FAISS, and generates an answer.
    """

    def __init__(
        self,
        *,
        config: Dict[str, Any],
        store: FAISSStore,
        embedder: EmbeddingPipeline,
        llm: LLM,
    ) -> None:
        self.config = config
        self.store = store
        self.embedder = embedder
        self.llm = llm
        self._logger = get_logger()

    @classmethod
    def from_config(cls, config_path: Optional[Path] = None, *, llm: Optional[LLM] = None) -> "RAGPipeline":
        load_dotenv()
        setup_logger()
        logger = get_logger()
        cfg = load_config(config_path)

        emb_model = str(cfg["embedding"]["model"])
        top_k = int(cfg["retrieval"]["top_k"])
        _ = top_k  # validated by access

        index_path = _resolve_path(str(cfg["vectorstore"]["index_path"]))
        docstore_path = _resolve_path(str(cfg["vectorstore"]["docstore_path"]))

        # all-MiniLM-L6-v2 is 384-dim; keep configurable later if needed.
        dim = int(cfg.get("embedding", {}).get("dim", 384))
        store = FAISSStore.load(index_path=index_path, docstore_path=docstore_path)
        if store.dim != dim:
            logger.info(f"Config dim={dim} differs from store dim={store.dim}; using store dim.")

        cache_path = _resolve_path("data/embedding_cache.sqlite")
        embedder = EmbeddingPipeline(model_name=emb_model, cache_path=cache_path, normalize_embeddings=True)

        if llm is None:
            llm_cfg = cfg["llm"]
            llm = get_llm(
                model_name=str(llm_cfg["model_name"]),
                max_new_tokens=int(llm_cfg.get("max_new_tokens", 512)),
                temperature=float(llm_cfg.get("temperature", 0.3)),
                do_sample=bool(llm_cfg.get("do_sample", True)),
            )

        return cls(config=cfg, store=store, embedder=embedder, llm=llm)

    def retrieve(self, question: str, *, top_k: Optional[int] = None) -> List[SourceChunk]:
        k = int(top_k if top_k is not None else self.config["retrieval"]["top_k"])
        q_emb = self.embedder.embed_query(question)
        hits = self.store.similarity_search(q_emb, k=k)
        return [SourceChunk(text=t, metadata=dict(m), score=float(s)) for t, m, s in hits]

    def answer(self, question: str, *, top_k: Optional[int] = None) -> Tuple[str, List[Dict[str, Any]]]:
        chunks = self.retrieve(question, top_k=top_k)
        context = _format_sources(chunks)
        prompt = _build_prompt(question, context)

        self._logger.info(f"Query received (top_k={len(chunks)}): {question}")
        raw = self.llm.invoke(prompt)
        answer = str(raw).strip()

        sources = [
            {
                "rank": i + 1,
                "score": ch.score,
                "metadata": ch.metadata,
                "text": ch.text,
            }
            for i, ch in enumerate(chunks)
        ]
        return answer, sources


_DEFAULT_PIPELINE: Optional[RAGPipeline] = None


def get_pipeline() -> RAGPipeline:
    global _DEFAULT_PIPELINE
    if _DEFAULT_PIPELINE is None:
        _DEFAULT_PIPELINE = RAGPipeline.from_config()
    return _DEFAULT_PIPELINE


def query_rag(question: str, top_k: int = 5) -> tuple[str, List[dict]]:
    """Convenience function used by API/UI."""
    pipeline = get_pipeline()
    answer, sources = pipeline.answer(question, top_k=top_k)
    return answer, sources
