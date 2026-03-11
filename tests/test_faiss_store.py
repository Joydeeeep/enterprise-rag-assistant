from __future__ import annotations

from pathlib import Path

import numpy as np

from vectorstore.faiss_store import FAISSStore


def test_faiss_store_add_search_save_load(tmp_path: Path) -> None:
    dim = 4
    store = FAISSStore(dim=dim, metric="ip", normalize_embeddings=True)

    texts = ["alpha", "beta", "gamma"]
    # Create deterministic embeddings where alpha is closest to query.
    embs = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    store.add_texts(texts=texts, embeddings=embs.tolist(), metadatas=[{"i": 0}, {"i": 1}, {"i": 2}])

    q = np.array([1.0, 0.1, 0.0, 0.0], dtype=np.float32)
    results = store.similarity_search(q.tolist(), k=2)
    assert len(results) == 2
    assert results[0][0] == "alpha"
    assert results[0][1]["i"] == 0

    index_path = tmp_path / "index.faiss"
    docstore_path = tmp_path / "docstore.pkl"
    store.save(index_path=index_path, docstore_path=docstore_path)

    loaded = FAISSStore.load(index_path=index_path, docstore_path=docstore_path)
    results2 = loaded.similarity_search(q.tolist(), k=2)
    assert results2[0][0] == "alpha"
    assert results2[0][1]["i"] == 0
