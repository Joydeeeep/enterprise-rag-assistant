from app.rag_pipeline import RAGPipeline
from evaluation.retrieval_evaluator import RetrievalEvaluator
from evaluation.logger import save_results


def main():
    print("Initializing RAG pipeline...")
    pipeline = RAGPipeline()

    print("Loading evaluator...")
    evaluator = RetrievalEvaluator(
        pipeline=pipeline,
        dataset_path="tests/eval_dataset.json"
    )

    print("Running evaluation...")
    results = evaluator.evaluate(k=5)

    recall = evaluator.compute_recall_at_k(results)

    print(f"\n📊 Recall@5: {recall:.2f}")

    save_results(results, recall)

    print("Results saved to evaluation/results.json")


if __name__ == "__main__":
    main()