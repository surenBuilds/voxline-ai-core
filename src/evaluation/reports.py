"""
Evaluation report generation.

Produces human-readable and machine-readable evaluation reports.
"""

import json
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

from src.evaluation.schemas import EvalReport, EvalRunConfig


def save_report(report: EvalReport, output_dir: str | Path) -> Path:
    """
    Save evaluation report to a directory.

    Creates:
        output_dir/results.json  — full case-by-case results
        output_dir/summary.json  — aggregated summary
        output_dir/config.json   — run configuration
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_dict = report.to_dict()

    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(report_dict, f, ensure_ascii=False, indent=2)

    summary = {
        "run_config": report_dict["run_config"],
        "summary": report_dict["summary"],
        "categories": report_dict["categories"],
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(report_dict["run_config"], f, ensure_ascii=False, indent=2)

    return output_dir


def load_report(report_dir: str | Path) -> EvalReport:
    """Load a saved evaluation report."""
    report_dir = Path(report_dir)
    results_path = report_dir / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"No results.json in {report_dir}")

    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    from src.evaluation.schemas import (
        CaseResult, CategorySummary, FailureCategory, HumanEvalScores,
        BenchmarkCategory,
    )

    run_config = EvalRunConfig(**{
        k: v for k, v in data["run_config"].items()
    })

    report = EvalReport(run_config=run_config)

    for r in data.get("results", []):
        fc = None
        if r.get("failure_category"):
            try:
                fc = FailureCategory(r["failure_category"])
            except ValueError:
                fc = FailureCategory.UNKNOWN

        human_scores = None
        if r.get("human_scores"):
            hs = r["human_scores"]
            human_scores = HumanEvalScores(
                coherence=hs.get("coherence"),
                relevance=hs.get("relevance"),
                correctness=hs.get("correctness"),
                instruction_following=hs.get("instruction_following"),
                language_quality=hs.get("language_quality"),
            )

        cr = CaseResult(
            case_id=r["case_id"],
            prompt=r["prompt"],
            actual_response=r["actual_response"],
            expected_answer=r.get("expected_answer"),
            metrics=r.get("metrics", {}),
            passed=r.get("passed"),
            failure_category=fc,
            failure_reason=r.get("failure_reason"),
            latency_ms=r.get("latency_ms", 0.0),
            token_count_input=r.get("token_count_input", 0),
            token_count_output=r.get("token_count_output", 0),
            human_scores=human_scores,
        )
        report.case_results.append(cr)

    report.compute_summary()
    return report


def format_report_text(report: EvalReport) -> str:
    """Format a human-readable evaluation report."""
    lines = []
    cfg = report.run_config

    lines.append("=" * 60)
    lines.append("VOXLINE MODEL EVALUATION")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Provider:     {cfg.provider_id}")
    lines.append(f"Model:        {cfg.model_id}")
    lines.append(f"Benchmark:    {cfg.benchmark_name} v{cfg.benchmark_version}")
    lines.append(f"Date:         {cfg.timestamp}")
    lines.append(f"Run ID:       {cfg.run_id}")
    lines.append(f"Temperature:  {cfg.generation_config.get('temperature', 'N/A')}")
    lines.append(f"Max tokens:   {cfg.generation_config.get('max_tokens', 'N/A')}")
    lines.append("")

    lines.append("-" * 60)
    lines.append("OVERALL RESULTS")
    lines.append("-" * 60)
    lines.append(f"Total cases:  {report.total_cases}")
    lines.append(f"Passed:       {report.total_passed}")
    lines.append(f"Failed:       {report.total_failed}")
    lines.append(f"Errors:       {report.total_errors}")
    lines.append(f"Pass rate:    {report.overall_pass_rate:.1%}")
    lines.append(f"Avg latency:  {report.overall_avg_latency_ms:.1f} ms")
    lines.append(f"Avg speed:    {report.overall_avg_throughput_tps:.1f} tokens/sec")
    lines.append("")

    if report.category_summaries:
        lines.append("-" * 60)
        lines.append("BY CATEGORY")
        lines.append("-" * 60)
        lines.append(f"{'Category':<30} {'Pass':>5} {'Total':>6} {'Rate':>7} {'Latency':>9}")
        lines.append("-" * 60)
        for cs in report.category_summaries:
            lines.append(
                f"{cs.category:<30} {cs.passed:>5} {cs.total_cases:>6} "
                f"{cs.pass_rate:>6.1%} {cs.avg_latency_ms:>8.1f}ms"
            )
        lines.append("")

    arm_results = [r for r in report.case_results if r.metrics.get("_language") == "hy"]
    en_results = [r for r in report.case_results if r.metrics.get("_language") == "en"]

    if arm_results and en_results:
        lines.append("-" * 60)
        lines.append("LANGUAGE BREAKDOWN")
        lines.append("-" * 60)
        arm_passed = sum(1 for r in arm_results if r.passed is True)
        en_passed = sum(1 for r in en_results if r.passed is True)
        lines.append(f"Armenian:     {arm_passed}/{len(arm_results)} ({arm_passed/len(arm_results):.1%})")
        lines.append(f"English:      {en_passed}/{len(en_results)} ({en_passed/len(en_results):.1%})")
        lines.append("")

    if report.failures:
        lines.append("-" * 60)
        lines.append(f"FAILED CASES ({len(report.failures)})")
        lines.append("-" * 60)
        for f in report.failures:
            lines.append(f"  [{f.case_id}] {f.failure_category.value if f.failure_category else 'unknown'}")
            lines.append(f"    Prompt:     {f.prompt[:100]}...")
            lines.append(f"    Expected:   {(f.expected_answer or '?')[:100]}")
            lines.append(f"    Got:        {f.actual_response[:100]}")
            if f.failure_reason:
                lines.append(f"    Reason:     {f.failure_reason[:150]}")
            lines.append("")

    lines.append("=" * 60)
    return "\n".join(lines)


def format_comparison_text(
    report_a: EvalReport,
    report_b: EvalReport,
    label_a: str = "A",
    label_b: str = "B",
) -> str:
    """Format a comparison between two evaluation runs."""
    lines = []

    lines.append("=" * 70)
    lines.append("EVALUATION COMPARISON")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"  {label_a}: {report_a.run_config.provider_id} / {report_a.run_config.model_id}")
    lines.append(f"  {label_b}: {report_b.run_config.provider_id} / {report_b.run_config.model_id}")
    lines.append("")

    def delta(a: float, b: float) -> str:
        d = b - a
        sign = "+" if d >= 0 else ""
        return f"{sign}{d:.1%}" if abs(d) < 10 else f"{sign}{d:.1f}"

    metrics = [
        ("Overall Pass Rate", report_a.overall_pass_rate, report_b.overall_pass_rate),
        ("Avg Latency (ms)", report_a.overall_avg_latency_ms, report_b.overall_avg_latency_ms),
        ("Avg Throughput (t/s)", report_a.overall_avg_throughput_tps, report_b.overall_avg_throughput_tps),
    ]

    arm_a = [r for r in report_a.case_results if r.metrics.get("_language") == "hy"]
    arm_b = [r for r in report_b.case_results if r.metrics.get("_language") == "hy"]
    en_a = [r for r in report_a.case_results if r.metrics.get("_language") == "en"]
    en_b = [r for r in report_b.case_results if r.metrics.get("_language") == "en"]

    if arm_a and arm_b:
        arm_rate_a = sum(1 for r in arm_a if r.passed) / len(arm_a)
        arm_rate_b = sum(1 for r in arm_b if r.passed) / len(arm_b)
        metrics.append(("Armenian Pass Rate", arm_rate_a, arm_rate_b))

    if en_a and en_b:
        en_rate_a = sum(1 for r in en_a if r.passed) / len(en_a)
        en_rate_b = sum(1 for r in en_b if r.passed) / len(en_b)
        metrics.append(("English Pass Rate", en_rate_a, en_rate_b))

    all_cats_a = {}
    all_cats_b = {}
    for cs in report_a.category_summaries:
        all_cats_a[cs.category] = cs.pass_rate
    for cs in report_b.category_summaries:
        all_cats_b[cs.category] = cs.pass_rate

    for cat in sorted(set(all_cats_a.keys()) | set(all_cats_b.keys())):
        rate_a = all_cats_a.get(cat, 0.0)
        rate_b = all_cats_b.get(cat, 0.0)
        metrics.append((f"  {cat}", rate_a, rate_b))

    header = f"{'Metric':<35} {label_a:>10} {label_b:>10} {'Delta':>10}"
    lines.append(header)
    lines.append("-" * 70)
    for name, va, vb in metrics:
        lines.append(f"{name:<35} {va:>9.1%} {vb:>9.1%} {delta(va, vb):>10}")

    lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)
