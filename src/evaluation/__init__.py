"""
Voxline AI Evaluation Framework.

Provides benchmarking, metrics, evaluation runner,
report generation, and provider comparison.
"""

from src.evaluation.schemas import (
    BenchmarkCase, BenchmarkCategory, CaseResult,
    EvalRunConfig, EvalReport, CategorySummary,
    MetricType, FailureCategory, HumanEvalScores,
)
from src.evaluation.metrics import compute_case_metrics, aggregate_category_metrics
from src.evaluation.datasets import (
    load_benchmark, save_benchmark, get_builtin_benchmarks,
    load_builtin_benchmark, filter_cases,
)
from src.evaluation.runner import EvaluationRunner, EvaluationError
from src.evaluation.reports import (
    save_report, load_report, format_report_text, format_comparison_text,
)
from src.evaluation.comparison import (
    compare_reports, detect_regression, load_and_compare, save_comparison,
    ComparisonResult, MetricDelta,
)

__all__ = [
    "BenchmarkCase", "BenchmarkCategory", "CaseResult",
    "EvalRunConfig", "EvalReport", "CategorySummary",
    "MetricType", "FailureCategory", "HumanEvalScores",
    "compute_case_metrics", "aggregate_category_metrics",
    "load_benchmark", "save_benchmark", "get_builtin_benchmarks",
    "load_builtin_benchmark", "filter_cases",
    "EvaluationRunner", "EvaluationError",
    "save_report", "load_report", "format_report_text", "format_comparison_text",
    "compare_reports", "detect_regression", "load_and_compare", "save_comparison",
    "ComparisonResult", "MetricDelta",
]
