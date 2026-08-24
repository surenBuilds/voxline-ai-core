"""
Benchmark dataset loading.

Loads benchmark cases from JSONL files.
Provides built-in benchmark datasets.
"""

import json
from pathlib import Path
from typing import List, Optional, Dict, Any

from src.evaluation.schemas import BenchmarkCase, BenchmarkCategory


BENCHMARK_DIR = Path(__file__).resolve().parent.parent.parent / "benchmarks"


def load_benchmark(path: str | Path) -> List[BenchmarkCase]:
    """Load benchmark cases from a JSONL file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {path}")
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                cases.append(BenchmarkCase.from_dict(data))
            except (json.JSONDecodeError, KeyError) as e:
                raise ValueError(f"Invalid benchmark data at line {line_num}: {e}")
    return cases


def save_benchmark(cases: List[BenchmarkCase], path: str | Path) -> None:
    """Save benchmark cases to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case.to_dict(), ensure_ascii=False) + "\n")


def get_builtin_benchmarks() -> Dict[str, Path]:
    """List available built-in benchmark files."""
    benchmarks = {}
    if BENCHMARK_DIR.exists():
        for f in sorted(BENCHMARK_DIR.glob("*.jsonl")):
            name = f.stem
            benchmarks[name] = f
    return benchmarks


def load_builtin_benchmark(name: str) -> List[BenchmarkCase]:
    """Load a built-in benchmark by name."""
    benchmarks = get_builtin_benchmarks()
    if name not in benchmarks:
        available = ", ".join(benchmarks.keys()) if benchmarks else "(none)"
        raise FileNotFoundError(
            f"Benchmark '{name}' not found. Available: {available}"
        )
    return load_benchmark(benchmarks[name])


def filter_cases(
    cases: List[BenchmarkCase],
    categories: Optional[List[str]] = None,
    languages: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
) -> List[BenchmarkCase]:
    """Filter benchmark cases by criteria."""
    filtered = cases
    if categories:
        cat_set = set(categories)
        filtered = [c for c in filtered if c.category.value in cat_set]
    if languages:
        lang_set = set(languages)
        filtered = [c for c in filtered if c.language in lang_set]
    if tags:
        tag_set = set(tags)
        filtered = [c for c in filtered if tag_set.intersection(c.tags)]
    return filtered
