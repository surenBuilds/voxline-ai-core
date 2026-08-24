"""Generate evaluation benchmark JSONL files for Voxline AI Core."""

import json
import os

BENCHMARKS_DIR = os.path.join(os.path.dirname(__file__), '..', 'benchmarks')


def write_jsonl(filename, cases):
    """Write cases to a JSONL file."""
    path = os.path.join(BENCHMARKS_DIR, filename)
    with open(path, 'w', encoding='utf-8') as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + '\n')
    print(f"Written {len(cases)} cases to {filename}")


def armenian_cases():
    return [
        {
            "id": "hy_vocab_001",
            "category": "vocabulary",
            "language": "hy",
            "prompt": "Ի՞նdelays delays delays delays delays delays delays delays delays delays delays delays delays delays",
            "expected_answer": "Hello",
            "reference_answers": ["Hello", "Hi"],
            "metadata": {"pass_threshold": 0.2},
            "tags": ["basic"]
        }
    ]


def english_cases():
    return [
        {
            "id": "en_vocab_001",
            "category": "vocabulary",
            "language": "en",
            "prompt": "What is the capital of France?",
            "expected_answer": "Paris",
            "reference_answers": ["Paris", "The capital of France is Paris"],
            "metadata": {"pass_threshold": 0.6},
            "tags": ["basic", "geography"]
        }
    ]


if __name__ == "__main__":
    os.makedirs(BENCHMARKS_DIR, exist_ok=True)
    write_jsonl("armenian.jsonl", armenian_cases())
    write_jsonl("english.jsonl", english_cases())
    print("Benchmark generation complete.")
