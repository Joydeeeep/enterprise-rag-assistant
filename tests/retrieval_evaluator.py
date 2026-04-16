from dataclasses import dataclass
from typing import List, Dict
import json


@dataclass
class EvalResult:
    query: str
    expected_docs: List[str]
    retrieved_docs: List[str]
    hit: int


class RetrievalEvaluator:
    def __init__(self, pipeline, dataset_path: str):
        self.pipeline = pipeline
        self.dataset = self._load_dataset(dataset_path)

    def _load_dataset(self, path: str) -> List[Dict]:
        with open(path, "r") as f:
            return json.load(f)

    def _extract_doc_names(self, results) -> List[str]:
        return [
            r.metadata.get("file", "").lower()
            for r in results
        ]

    def _check_hit(self, retrieved_docs: List[str], expected_docs: List[str]) -> int:
        # if no expected docs → success only if nothing relevant retrieved
        if not expected_docs:
            return 1 if len(retrieved_docs) == 0 else 0

        for expected in expected_docs:
            for doc in retrieved_docs:
                if expected.lower() in doc:
                    return 1
        return 0

    def evaluate(self, k: int = 5) -> List[EvalResult]:
        results: List[EvalResult] = []

        for sample in self.dataset:
            query = sample["query"]
            expected_docs = sample["expected_docs"]

            retrieved = self.pipeline.vectorstore.similarity_search(query, k=k)
            retrieved_docs = self._extract_doc_names(retrieved)

            hit = self._check_hit(retrieved_docs, expected_docs)

            results.append(
                EvalResult(
                    query=query,
                    expected_docs=expected_docs,
                    retrieved_docs=retrieved_docs,
                    hit=hit,
                )
            )

        return results

    def compute_recall_at_k(self, results: List[EvalResult]) -> float:
        hits = sum(r.hit for r in results)
        return hits / len(results)