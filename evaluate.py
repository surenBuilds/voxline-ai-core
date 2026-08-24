#!/usr/bin/env python3
"""Voxline AI Evaluation CLI.

Run benchmark evaluations against a model provider.

Usage:
    python evaluate.py --provider qwen --benchmark benchmarks/english.jsonl
    python evaluate.py --provider native --benchmark benchmarks/armenian.jsonl --categories vocabulary
    python evaluate.py --provider qwen --benchmark benchmarks/english.jsonl --output eval_results/
    python evaluate.py --list-benchmarks
"""

import sys
import argparse
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.evaluation.datasets import load_benchmark, get_builtin_benchmarks, filter_cases
from src.evaluation.runner import EvaluationRunner
from src.evaluation.reports import save_report, format_report_text
from src.providers.base import GenerationConfig


def list_benchmarks():
    benchmarks = get_builtin_benchmarks()
    if not benchmarks:
        print("No benchmark files found in benchmarks/")
        return
    print("Available built-in benchmarks:")
    for name, path in benchmarks.items():
        cases = load_benchmark(path)
        langs = set(c.language for c in cases)
        cats = set(c.category.value for c in cases)
        print(f"  {name}: {len(cases)} cases, languages={langs}, categories={cats}")


def create_provider(provider_name):
    if provider_name == "native":
        from src.providers.local_voxline import LocalVoxlineProvider
        from src.model.transformer import VoxlineTransformer
        from src.tokenizer.bpe import BPETokenizer
        from src.config.model_config import ModelConfig
        import torch
        import yaml

        config_path = Path("configs/model_configs.yaml")
        with open(config_path) as f:
            model_config_dict = yaml.safe_load(f)["v0_4_small"]["model"]

        checkpoint_dir = Path("checkpoints/v0_4")
        tokenizer = BPETokenizer(vocab_size=model_config_dict["vocab_size"])
        tokenizer.load(str(checkpoint_dir / "tokenizer.json"))

        model = VoxlineTransformer(
            vocab_size=tokenizer.get_vocab_size(),
            d_model=model_config_dict["d_model"],
            num_layers=model_config_dict["num_layers"],
            num_heads=model_config_dict["num_heads"],
            d_ff=model_config_dict["d_ff"],
            max_seq_len=model_config_dict["max_seq_len"],
            dropout=model_config_dict["dropout"],
        )
        checkpoint = torch.load(
            checkpoint_dir / "best_model.pt", map_location="cpu", weights_only=False
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        model_config = ModelConfig(
            model_type="voxline_transformer",
            model_version="0.4.0",
            vocab_size=tokenizer.get_vocab_size(),
            d_model=model_config_dict["d_model"],
            max_seq_len=model_config_dict["max_seq_len"],
            num_layers=model_config_dict["num_layers"],
            num_heads=model_config_dict["num_heads"],
            d_ff=model_config_dict["d_ff"],
            dropout=model_config_dict["dropout"],
        )
        return LocalVoxlineProvider(model, tokenizer, model_config, device="cpu")

    elif provider_name == "qwen":
        from src.providers.qwen_provider import QwenProvider
        model_path = "models/Qwen2.5-0.5B-Instruct"
        if not Path(model_path).exists():
            print(f"Error: Qwen model not found at {model_path}")
            sys.exit(1)
        return QwenProvider(model_path=model_path, device="cpu")

    else:
        print(f"Error: Unknown provider '{provider_name}'. Available: native, qwen")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Voxline AI Evaluation CLI")
    parser.add_argument(
        "--provider",
        choices=["native", "qwen"],
        default="qwen",
        help="AI provider to evaluate (default: qwen)",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        help="Path to benchmark JSONL file, or built-in benchmark name",
    )
    parser.add_argument(
        "--list-benchmarks",
        action="store_true",
        help="List available built-in benchmarks",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        help="Filter by category (e.g. vocabulary reasoning)",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        help="Filter by language (hy en)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=150,
        help="Max tokens per response (default: 150)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Generation temperature (default: 0.7)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output directory for evaluation results",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Timeout per case in seconds (default: 60)",
    )
    args = parser.parse_args()

    if args.list_benchmarks:
        list_benchmarks()
        return

    if not args.benchmark:
        parser.error("--benchmark is required (or use --list-benchmarks)")

    benchmark_path = Path(args.benchmark)
    if not benchmark_path.exists():
        builtin = get_builtin_benchmarks()
        if args.benchmark in builtin:
            benchmark_path = builtin[args.benchmark]
        else:
            print(f"Error: Benchmark file not found: {args.benchmark}")
            sys.exit(1)

    cases = load_benchmark(benchmark_path)
    if args.categories or args.languages:
        cases = filter_cases(cases, categories=args.categories, languages=args.languages)

    if not cases:
        print("Error: No benchmark cases matched the given filters")
        sys.exit(1)

    print(f"Loading provider: {args.provider}")
    provider = create_provider(args.provider)

    gen_config = GenerationConfig(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        do_sample=True,
        top_p=0.9,
    )

    runner = EvaluationRunner(provider, gen_config, timeout_seconds=args.timeout)

    print(f"Running evaluation: {len(cases)} cases from {benchmark_path.name}")
    print(f"  max_tokens={args.max_tokens}, temperature={args.temperature}")
    print()

    report = runner.run_cases(cases)

    report_text = format_report_text(report)
    print(report_text)

    if args.output:
        output_dir = Path(args.output)
        save_report(report, output_dir)
        print(f"\nResults saved to {output_dir}/")
        print(f"  results.json  — full case-by-case results")
        print(f"  summary.json  — aggregated summary")
        print(f"  config.json   — run configuration")


if __name__ == "__main__":
    main()
