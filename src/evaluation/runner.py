"""
Evaluation runner.

Orchestrates evaluation of an AIProvider against a benchmark suite.
Records responses, measures latency, computes metrics, produces reports.
"""

import asyncio
import time
import logging
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

from src.evaluation.schemas import (
    BenchmarkCase, CaseResult, EvalRunConfig, EvalReport,
    FailureCategory, HumanEvalScores, EvaluationStatus,
)
from src.evaluation.metrics import compute_case_metrics
from src.evaluation.datasets import load_benchmark, filter_cases
from src.providers.base import AIProvider, GenerationConfig
from src.errors import VoxlineError


logger = logging.getLogger(__name__)


class EvaluationError(VoxlineError):
    """Error during evaluation execution."""


class EvaluationRunner:
    """
    Runs benchmark evaluations against an AIProvider.

    Usage:
        runner = EvaluationRunner(provider, gen_config)
        report = runner.run("benchmarks/armenian.jsonl")
    """

    def __init__(
        self,
        provider: AIProvider,
        generation_config: Optional[GenerationConfig] = None,
        timeout_seconds: float = 60.0,
    ):
        self.provider = provider
        self.gen_config = generation_config or GenerationConfig(
            max_tokens=150,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
        )
        self.timeout_seconds = timeout_seconds

    def run(
        self,
        benchmark_path: str | Path,
        categories: Optional[List[str]] = None,
        languages: Optional[List[str]] = None,
    ) -> EvalReport:
        """
        Run evaluation on a benchmark file.

        Args:
            benchmark_path: Path to JSONL benchmark
            categories: Optional category filter
            languages: Optional language filter ("hy", "en")

        Returns:
            EvalReport with all results
        """
        cases = load_benchmark(benchmark_path)
        if categories or languages:
            cases = filter_cases(cases, categories=categories, languages=languages)

        if not cases:
            raise EvaluationError("No benchmark cases matched the given filters")

        run_config = EvalRunConfig(
            provider_id=self.provider.provider_id,
            model_id=self.provider.model_id,
            benchmark_name=Path(benchmark_path).stem,
            generation_config={
                "max_tokens": self.gen_config.max_tokens,
                "temperature": self.gen_config.temperature,
                "top_p": self.gen_config.top_p,
                "top_k": self.gen_config.top_k,
                "do_sample": self.gen_config.do_sample,
            },
            environment={
                "python": sys.version.split()[0],
                "platform": sys.platform,
            },
        )

        report = EvalReport(run_config=run_config)
        total = len(cases)

        for i, case in enumerate(cases):
            logger.info(f"[{i+1}/{total}] {case.id} ({case.category.value}, {case.language})")
            case_result = self._evaluate_case(case)
            case_result.metrics["_category"] = case.category.value
            case_result.metrics["_language"] = case.language
            report.case_results.append(case_result)

        report.compute_summary()
        return report

    def run_cases(self, cases: List[BenchmarkCase]) -> EvalReport:
        """Run evaluation on pre-loaded cases."""
        run_config = EvalRunConfig(
            provider_id=self.provider.provider_id,
            model_id=self.provider.model_id,
            benchmark_name="inline",
            generation_config={
                "max_tokens": self.gen_config.max_tokens,
                "temperature": self.gen_config.temperature,
                "top_p": self.gen_config.top_p,
            },
        )
        report = EvalReport(run_config=run_config)
        for case in cases:
            case_result = self._evaluate_case(case)
            case_result.metrics["_category"] = case.category.value
            case_result.metrics["_language"] = case.language
            report.case_results.append(case_result)
        report.compute_summary()
        return report

    def _evaluate_case(self, case: BenchmarkCase) -> CaseResult:
        """Evaluate a single benchmark case."""
        result = CaseResult(
            case_id=case.id,
            prompt=case.prompt,
            actual_response="",
            expected_answer=case.expected_answer,
        )

        start_time = time.time()
        try:
            response = asyncio.run(self._get_response(case))
            elapsed_ms = (time.time() - start_time) * 1000

            result.actual_response = response
            result.latency_ms = elapsed_ms
            result.token_count_output = len(response.split())
            result.token_count_input = len(case.prompt.split())

            result.metrics = compute_case_metrics(
                actual=response,
                expected=case.expected_answer,
                references=case.reference_answers,
                category=case.category.value,
                metric_config=case.metadata.get("metric_config"),
            )

            result.passed = self._determine_pass(result.metrics, case)

            if not result.passed and result.passed is not None:
                result.failure_category = self._classify_failure(result, case)
                result.failure_reason = self._explain_failure(result, case)

        except asyncio.TimeoutError:
            result.latency_ms = self.timeout_seconds * 1000
            result.passed = None
            result.failure_category = FailureCategory.TIMEOUT
            result.failure_reason = f"Timed out after {self.timeout_seconds}s"
        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            result.latency_ms = elapsed_ms
            result.actual_response = f"[Error: {str(e)}]"
            result.passed = None
            result.failure_category = FailureCategory.PROVIDER_ERROR
            result.failure_reason = str(e)

        return result

    async def _get_response(self, case: BenchmarkCase) -> str:
        """Get response from provider with timeout."""
        if case.conversation_history:
            messages = case.conversation_history + [
                {"role": "user", "content": case.prompt}
            ]
            return await asyncio.wait_for(
                self.provider.chat(messages, self.gen_config),
                timeout=self.timeout_seconds,
            )
        else:
            return await asyncio.wait_for(
                self.provider.generate(case.prompt, self.gen_config),
                timeout=self.timeout_seconds,
            )

    def _determine_pass(self, metrics: Dict[str, float], case: BenchmarkCase) -> bool:
        """
        Determine if a case passed based on metrics and metadata.

        Uses task-specific metrics when available, falls back to generic scoring.
        """
        threshold = case.metadata.get("pass_threshold", 0.5)

        # Exact match always passes
        if "exact_match" in metrics and metrics["exact_match"] >= 1.0:
            return True

        # Normalized match always passes
        if "normalized_match" in metrics and metrics["normalized_match"] >= 1.0:
            return True

        # Number match always passes
        if "number_match" in metrics and metrics["number_match"] >= 1.0:
            return True

        # Classification accuracy always passes
        if "classification_accuracy" in metrics and metrics["classification_accuracy"] >= 1.0:
            return True

        # Task-specific primary metric checks
        task_primary = {
            "vocabulary": "vocabulary_accuracy",
            "sentence_completion": "sentence_completion_match",
            "question_answering": "qa_match",
            "instruction_following": "instruction_following_score",
            "translation": "translation_score",
            "reasoning": "reasoning_score",
        }
        category = case.category.value if case.category else ""
        if category in task_primary:
            key = task_primary[category]
            if key in metrics and metrics[key] >= threshold:
                return True

        # Keyword match passes if >= 0.8
        if "keyword_match" in metrics and metrics["keyword_match"] >= 0.8:
            return True

        # Smart contains passes if >= threshold
        if "smart_contains" in metrics and metrics["smart_contains"] >= threshold:
            return True

        # Any reference smart contains passes
        if "any_smart_contains" in metrics and metrics["any_smart_contains"] >= threshold:
            return True

        # Generic similarity fallback
        sim_scores = [
            metrics.get("sequence_similarity", 0),
            metrics.get("best_reference_similarity", 0),
            metrics.get("word_overlap", 0),
        ]
        max_sim = max(sim_scores) if sim_scores else 0.0

        if "format_check" in metrics and metrics["format_check"] < 1.0:
            return False

        if case.expected_answer:
            return max_sim >= threshold

        if case.reference_answers:
            return max_sim >= threshold

        return len(case.actual_response.strip()) > 0

    def _classify_failure(self, result: CaseResult, case: BenchmarkCase) -> FailureCategory:
        """Classify the type of failure."""
        if result.failure_category:
            return result.failure_category

        if not result.actual_response.strip():
            return FailureCategory.GENERATION_FAILURE

        if result.metrics.get("format_check", 1.0) < 0.5:
            return FailureCategory.FORMATTING_FAILURE

        if case.conversation_history:
            return FailureCategory.CONTEXT_FAILURE

        if case.category.value in ("reasoning", "question_answering"):
            if result.metrics.get("exact_match", 0) == 0:
                return FailureCategory.REASONING_FAILURE

        if case.language in ("hy", "en"):
            if result.metrics.get("sequence_similarity", 1.0) < 0.1:
                return FailureCategory.LANGUAGE_ERROR

        return FailureCategory.FACTUAL_ERROR

    def _explain_failure(self, result: CaseResult, case: BenchmarkCase) -> str:
        """Generate human-readable failure explanation."""
        parts = []
        if result.failure_category == FailureCategory.FORMATTING_FAILURE:
            parts.append("Response does not match expected format")
        elif result.failure_category == FailureCategory.CONTEXT_FAILURE:
            parts.append("Response does not retain conversation context")
        elif result.failure_category == FailureCategory.REASONING_FAILURE:
            parts.append("Reasoning result does not match expected answer")
        elif result.failure_category == FailureCategory.LANGUAGE_ERROR:
            parts.append("Response quality too low or wrong language")
        elif result.failure_category == FailureCategory.GENERATION_FAILURE:
            parts.append("Model produced empty or near-empty response")
        else:
            parts.append(f"Expected: {case.expected_answer}")
            parts.append(f"Got: {result.actual_response[:200]}")

        if result.metrics:
            top_metrics = sorted(
                [(k, v) for k, v in result.metrics.items() if not k.startswith("_")],
                key=lambda x: x[1], reverse=True
            )[:3]
            parts.append(
                "Metrics: " + ", ".join(f"{k}={v:.2f}" for k, v in top_metrics)
            )

        return "; ".join(parts)
