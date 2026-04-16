import json
from datetime import datetime


def save_results(results, recall, output_path="evaluation/results.json"):
    data = {
        "timestamp": datetime.utcnow().isoformat(),
        "recall_at_k": recall,
        "results": [
            {
                "query": r.query,
                "expected": r.expected_docs,
                "retrieved": r.retrieved_docs,
                "hit": r.hit,
            }
            for r in results
        ],
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)