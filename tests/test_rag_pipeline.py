from __future__ import annotations

from typing import Any, Dict, List, Tuple

from app.rag_pipeline import RAGPipeline
from vectorstore.faiss_store import FAISSStore


class FakeEmbedder:
    def __init__(self, mapping: Dict[str, List[float]]) -> None:
        self.mapping = mapping

    def embed_query(self, text: str) -> List[float]:
        return self.mapping[text]


class FakeLLM:
    def __init__(self) -> None:
        self.prompts: List[str] = []

    def invoke(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return "FAKE_ANSWER"


def test_rag_pipeline_retrieve_and_answer() -> None:
    dim = 4
    store = FAISSStore(dim=dim, metric="ip", normalize_embeddings=True)
    store.add_texts(
        texts=["policy about PTO", "security guidelines"],
        embeddings=[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        metadatas=[
            {"source": "doc1.txt", "section": "hr"},
            {"source": "doc2.txt", "section": "security"},
        ],
    )

    embedder = FakeEmbedder({"What is PTO policy?": [1.0, 0.1, 0.0, 0.0]})
    llm = FakeLLM()

    cfg: Dict[str, Any] = {
        "retrieval": {"top_k": 1},
    }
    pipeline = RAGPipeline(config=cfg, store=store, embedder=embedder, llm=llm)  # type: ignore[arg-type]

    chunks = pipeline.retrieve("What is PTO policy?")
    assert len(chunks) == 1
    assert "PTO" in chunks[0].text
    assert chunks[0].metadata["source"] == "doc1.txt"

    answer, sources = pipeline.answer("What is PTO policy?")
    assert answer == "FAKE_ANSWER"
    assert len(sources) == 1
    assert sources[0]["metadata"]["section"] == "hr"
    assert "Context" in llm.prompts[-1]
