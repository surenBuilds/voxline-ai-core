#!/usr/bin/env python3
"""Voxline AI Evaluation Comparison CLI.

Compare two evaluation runs and detect regressions.

Usage:
    python compare_evaluations.py run_a/ run_b/
    python compare_evaluations.py run_a/ run_b/ --threshold 0.05
    python compare_evaluations.py run_a/ run_b/ --output comparison.json
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.evaluation.comparison import (
    load_and_compare, detect_regression, save_comparison,
)
from src.evaluation.reports import load_report, format_comparison_text


def main():
    parser = argparse.ArgumentParser(
        description="Compare two Voxline evaluation runs"
    )
    parser.add_argument(
        "run_a",
        type=str,
        help="Path to first evaluation results directory (baseline)",
    )
    parser.add_argument(
        "run_b",
        type=str,
        help="Path to second evaluation results directory (current)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Regression detection threshold (default: 0.05)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Save comparison results to JSON file",
    )
    args = parser.parse_args()

    run_a_path = Path(args.run_a)
    run_b_path = Path(args.run_b)

    if not run_a_path.exists():
        print(f"Error: Run A directory not found: {run_a_path}")
        sys.exit(1)
    if not run_b_path.exists():
        print(f"Error: Run B directory not found: {run_b_path}")
        sys.exit(1)

    print("Loading reports...")
    report_a = load_report(run_a_path)
    report_b = load_report(run_b_path)

    comparison = load_and_compare(run_a_path, run_b_path)

    label_a = f"{report_a.run_config.provider_id}/{report_a.run_config.model_id}"
    label_b = f"{report_b.run_config.provider_id}/{report_b.run_config.model_id}"

    print()
    print(format_comparison_text(report_a, report_b, label_a, label_b))

    if comparison.regressions:
        print()
        print(f"REGRESSIONS DETECTED ({len(comparison.regressions)}):")
        for r in comparison.regressions:
            direction = "worse" if r.direction == "higher_is_better" else "higher"
            print(f"  {r.metric_name}: {r.value_a:.3f} -> {r.value_b:.3f} ({r.delta:+.3f} {direction})")
    else:
        print()
        print("No regressions detected.")

    significant = detect_regression(report_a, report_b, threshold=args.threshold)
    if significant:
        print()
        print(f"SIGNIFICANT REGRESSIONS (threshold={args.threshold}):")
        for r in significant:
            print(f"  {r.metric_name}: {r.value_a:.3f} -> {r.value_b:.3f} ({r.delta:+.3f})")
        sys.exit(1)
    else:
        print(f"\nNo regressions exceed threshold ({args.threshold}).")

    if args.output:
        output_path = save_comparison(comparison, args.output)
        print(f"\nComparison saved to {output_path}")


if __name__ == "__main__":
    main()
