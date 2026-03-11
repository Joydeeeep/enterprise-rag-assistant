"""
Embedding pipeline: convert text chunks into embeddings (SentenceTransformers).
"""

import hashlib
import sqlite3
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from utils.logger import get_logger

class EmbeddingPipeline:
    """Generate embeddings for text chunks."""

    def __init__(
        self,
        *,
        model_name: str,
        cache_path: Optional[Path] = None,
        normalize_embeddings: bool = True,
        batch_size: int = 64,
    ) -> None:
        """
        Args:
            model_name: SentenceTransformers model name.
            cache_path: Optional SQLite cache path to avoid repeated embeddings.
            normalize_embeddings: If True, L2-normalize vectors (good for cosine similarity).
            batch_size: Encode batch size.
        """
        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name)
        self._logger = get_logger()

        self.cache_path = Path(cache_path) if cache_path is not None else None
        if self.cache_path is not None:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_cache()

    def _connect(self) -> sqlite3.Connection:
        if self.cache_path is None:
            raise RuntimeError("Cache is disabled")
        conn = sqlite3.connect(str(self.cache_path))
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_cache(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    model TEXT NOT NULL,
                    text_hash TEXT NOT NULL,
                    dim INTEGER NOT NULL,
                    vec BLOB NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    PRIMARY KEY (model, text_hash)
                );
                """
            )

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()

    def _fetch_cached(self, hashes: Sequence[str]) -> dict[str, np.ndarray]:
        if self.cache_path is None or not hashes:
            return {}

        # SQLite has a variable limit; query in chunks.
        out: dict[str, np.ndarray] = {}
        chunk_size = 500
        with self._connect() as conn:
            for i in range(0, len(hashes), chunk_size):
                hs = hashes[i : i + chunk_size]
                placeholders = ",".join(["?"] * len(hs))
                rows = conn.execute(
                    f"""
                    SELECT text_hash, dim, vec
                    FROM embeddings
                    WHERE model = ? AND text_hash IN ({placeholders})
                    """,
                    (self.model_name, *hs),
                ).fetchall()
                for text_hash, dim, vec in rows:
                    arr = np.frombuffer(vec, dtype=np.float32).reshape((dim,))
                    out[str(text_hash)] = arr
        return out

    def _store_cached(self, items: Iterable[Tuple[str, np.ndarray]]) -> None:
        if self.cache_path is None:
            return
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO embeddings (model, text_hash, dim, vec)
                VALUES (?, ?, ?, ?)
                """,
                [
                    (self.model_name, h, int(v.shape[0]), sqlite3.Binary(v.astype(np.float32).tobytes()))
                    for h, v in items
                ],
            )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Convert list of texts to embedding vectors.

        Uses an optional SQLite cache keyed by sha256(text) to avoid recomputation.

        Returns:
            List of embeddings (each embedding is a list[float]).
        """
        if not texts:
            return []

        hashes = [self._hash_text(t) for t in texts]
        cached = self._fetch_cached(hashes)

        missing_idxs = [i for i, h in enumerate(hashes) if h not in cached]
        if missing_idxs:
            self._logger.info(f"Embedding {len(missing_idxs)} new texts (cache hit {len(texts) - len(missing_idxs)})")
            missing_texts = [texts[i] for i in missing_idxs]
            vecs = self._model.encode(
                missing_texts,
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize_embeddings,
                convert_to_numpy=True,
                show_progress_bar=False,
            ).astype(np.float32)
            self._store_cached((hashes[i], vecs[j]) for j, i in enumerate(missing_idxs))
            for j, i in enumerate(missing_idxs):
                cached[hashes[i]] = vecs[j]
        else:
            self._logger.info(f"Embedding cache hit for {len(texts)} texts")

        ordered = np.stack([cached[h] for h in hashes], axis=0)
        return ordered.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string."""
        return self.embed_documents([text])[0]
