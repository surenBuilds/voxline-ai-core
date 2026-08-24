"""
Evaluation comparison and regression detection.

Compares two evaluation runs and detects performance regressions.
"""

import json
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass

from src.evaluation.schemas import EvalReport
from src.evaluation.reports import load_report


@dataclass
class MetricDelta:
    """A single metric difference between two runs."""
    metric_name: str
    value_a: float
    value_b: float
    delta: float
    direction: str  # "higher_is_better" or "lower_is_better"
    regressed: bool


@dataclass
class ComparisonResult:
    """Result of comparing two evaluation runs."""
    run_a_id: str
    run_b_id: str
    provider_a: str
    provider_b: str
    model_a: str
    model_b: str
    overall_delta: float
    deltas: List[MetricDelta]
    regressions: List[MetricDelta]
    improvements: List[MetricDelta]


def compare_reports(
    report_a: EvalReport,
    report_b: EvalReport,
) -> ComparisonResult:
    """Compare two evaluation reports and identify regressions."""
    deltas: List[MetricDelta] = []

    metric_defs = [
        ("overall_pass_rate", "higher_is_better"),
        ("overall_avg_latency_ms", "lower_is_better"),
        ("overall_avg_throughput_tps", "higher_is_better"),
    ]

    pairs = [
        (getattr(report_a, name), getattr(report_b, name))
        for name, _ in metric_defs
    ]

    for (name, direction), (va, vb) in zip(metric_defs, pairs):
        delta_val = vb - va
        regressed = (
            (direction == "higher_is_better" and delta_val < -0.01) or
            (direction == "lower_is_better" and delta_val > 0.01)
        )
        deltas.append(MetricDelta(
            metric_name=name,
            value_a=va,
            value_b=vb,
            delta=delta_val,
            direction=direction,
            regressed=regressed,
        ))

    cats_a = {cs.category: cs.pass_rate for cs in report_a.category_summaries}
    cats_b = {cs.category: cs.pass_rate for cs in report_b.category_summaries}
    all_cats = sorted(set(cats_a.keys()) | set(cats_b.keys()))

    for cat in all_cats:
        va = cats_a.get(cat, 0.0)
        vb = cats_b.get(cat, 0.0)
        delta_val = vb - va
        regressed = delta_val < -0.01
        deltas.append(MetricDelta(
            metric_name=f"category/{cat}",
            value_a=va,
            value_b=vb,
            delta=delta_val,
            direction="higher_is_better",
            regressed=regressed,
        ))

    regressions = [d for d in deltas if d.regressed]
    improvements = [d for d in deltas if d.delta > 0.01 and d.direction == "higher_is_better"
                    or d.delta < -0.01 and d.direction == "lower_is_better"]

    overall_delta = report_b.overall_pass_rate - report_a.overall_pass_rate

    return ComparisonResult(
        run_a_id=report_a.run_config.run_id,
        run_b_id=report_b.run_config.run_id,
        provider_a=report_a.run_config.provider_id,
        provider_b=report_b.run_config.provider_id,
        model_a=report_a.run_config.model_id,
        model_b=report_b.run_config.model_id,
        overall_delta=overall_delta,
        deltas=deltas,
        regressions=regressions,
        improvements=improvements,
    )


def detect_regression(
    baseline: EvalReport,
    current: EvalReport,
    threshold: float = 0.05,
) -> List[MetricDelta]:
    """
    Detect regressions exceeding a threshold.

    Returns list of metrics that regressed by more than threshold.
    """
    result = compare_reports(baseline, current)
    return [
        d for d in result.regressions
        if abs(d.delta) > threshold
    ]


def load_and_compare(
    path_a: str | Path,
    path_b: str | Path,
) -> ComparisonResult:
    """Load two saved reports and compare them."""
    report_a = load_report(path_a)
    report_b = load_report(path_b)
    return compare_reports(report_a, report_b)


def save_comparison(comparison: ComparisonResult, output_path: str | Path) -> Path:
    """Save comparison results to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "run_a_id": comparison.run_a_id,
        "run_b_id": comparison.run_b_id,
        "provider_a": comparison.provider_a,
        "provider_b": comparison.provider_b,
        "model_a": comparison.model_a,
        "model_b": comparison.model_b,
        "overall_delta": comparison.overall_delta,
        "deltas": [
            {
                "metric_name": d.metric_name,
                "value_a": d.value_a,
                "value_b": d.value_b,
                "delta": d.delta,
                "direction": d.direction,
                "regressed": d.regressed,
            }
            for d in comparison.deltas
        ],
        "regressions_count": len(comparison.regressions),
        "improvements_count": len(comparison.improvements),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return output_path
