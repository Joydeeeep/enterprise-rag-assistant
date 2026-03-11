"""
FAISS vector store: persist and reload index; similarity search.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import faiss
import numpy as np

from utils.logger import get_logger

class FAISSStore:
    """FAISS index with optional persistence and docstore."""

    def __init__(
        self,
        *,
        dim: int,
        metric: str = "ip",
        normalize_embeddings: bool = True,
        index: Optional[faiss.Index] = None,
        docstore: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Args:
            dim: Embedding dimension.
            metric: 'ip' (inner product) or 'l2'. Use 'ip' for cosine if normalized.
            normalize_embeddings: If True, L2-normalize vectors before adding/searching.
            index: Optional prebuilt FAISS index (used for load()).
            docstore: Optional docstore list mapping vector id -> {text, metadata}.
        """
        self.dim = int(dim)
        self.metric = metric
        self.normalize_embeddings = normalize_embeddings
        self._logger = get_logger()

        if index is None:
            if metric == "ip":
                self.index = faiss.IndexFlatIP(self.dim)
            elif metric == "l2":
                self.index = faiss.IndexFlatL2(self.dim)
            else:
                raise ValueError("metric must be one of: 'ip', 'l2'")
        else:
            self.index = index

        self.docstore: List[Dict[str, Any]] = docstore or []

    @staticmethod
    def _as_matrix(vectors: Sequence[Sequence[float]], dim: int) -> np.ndarray:
        mat = np.asarray(vectors, dtype=np.float32)
        if mat.ndim != 2 or mat.shape[1] != dim:
            raise ValueError(f"Expected shape (n, {dim}), got {mat.shape}")
        return mat

    def _maybe_normalize(self, mat: np.ndarray) -> np.ndarray:
        if not self.normalize_embeddings:
            return mat
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return mat / norms

    def add_texts(
        self,
        *,
        texts: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> None:
        """
        Add texts + embeddings to the FAISS store.

        Args:
            texts: Raw chunk texts.
            embeddings: Corresponding vectors.
            metadatas: Optional metadata dict per text.
        """
        if len(texts) == 0:
            return
        if len(texts) != len(embeddings):
            raise ValueError("texts and embeddings must have the same length")
        if metadatas is not None and len(metadatas) != len(texts):
            raise ValueError("metadatas must match texts length")

        mat = self._as_matrix(embeddings, self.dim)
        mat = self._maybe_normalize(mat)

        start_id = len(self.docstore)
        self.index.add(mat)

        for i, text in enumerate(texts):
            md = dict(metadatas[i]) if metadatas is not None else {}
            self.docstore.append({"text": text, "metadata": md})

        self._logger.info(f"Added {len(texts)} vectors to FAISS (ids {start_id}..{len(self.docstore)-1})")

    def add_embeddings(self, embeddings: List[List[float]], metadatas: List[dict]) -> None:
        """
        Backwards-compatible add method.

        Expects each metadata dict to contain a 'text' field; that text is stored in the docstore.
        Prefer `add_texts()` for explicit inputs.
        """
        texts: List[str] = []
        cleaned_metas: List[Dict[str, Any]] = []
        for md in metadatas:
            if "text" not in md:
                raise ValueError("add_embeddings expects each metadata to include a 'text' field")
            text = str(md["text"])
            md2 = dict(md)
            md2.pop("text", None)
            texts.append(text)
            cleaned_metas.append(md2)
        self.add_texts(texts=texts, embeddings=embeddings, metadatas=cleaned_metas)

    def similarity_search(self, query_embedding: List[float], k: int) -> List[Tuple[str, dict, float]]:
        """Return top-k (text, metadata, score) for query embedding."""
        if k <= 0:
            return []
        if self.index.ntotal == 0:
            return []

        q = np.asarray([query_embedding], dtype=np.float32)
        if q.ndim != 2 or q.shape[1] != self.dim:
            raise ValueError(f"Expected query dim {self.dim}, got {q.shape}")
        q = self._maybe_normalize(q)

        scores, idxs = self.index.search(q, min(k, self.index.ntotal))
        results: List[Tuple[str, dict, float]] = []
        for score, idx in zip(scores[0].tolist(), idxs[0].tolist()):
            if idx < 0 or idx >= len(self.docstore):
                continue
            item = self.docstore[idx]
            results.append((str(item["text"]), dict(item.get("metadata") or {}), float(score)))
        return results

    def save(self, index_path: Path, docstore_path: Path) -> None:
        """Persist index and docstore to disk."""
        index_path = Path(index_path)
        docstore_path = Path(docstore_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        docstore_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(index_path))
        with open(docstore_path, "wb") as f:
            pickle.dump(
                {
                    "dim": self.dim,
                    "metric": self.metric,
                    "normalize_embeddings": self.normalize_embeddings,
                    "docstore": self.docstore,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        self._logger.info(f"Saved FAISS index to {index_path} and docstore to {docstore_path}")

    @classmethod
    def load(cls, index_path: Path, docstore_path: Path) -> "FAISSStore":
        """Load index and docstore from disk."""
        index_path = Path(index_path)
        docstore_path = Path(docstore_path)
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        if not docstore_path.exists():
            raise FileNotFoundError(f"Docstore not found: {docstore_path}")

        index = faiss.read_index(str(index_path))
        with open(docstore_path, "rb") as f:
            payload = pickle.load(f)

        store = cls(
            dim=int(payload["dim"]),
            metric=str(payload.get("metric", "ip")),
            normalize_embeddings=bool(payload.get("normalize_embeddings", True)),
            index=index,
            docstore=list(payload.get("docstore") or []),
        )
        store._logger.info(f"Loaded FAISS index (ntotal={store.index.ntotal}) from {index_path}")
        return store
