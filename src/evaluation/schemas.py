"""
Evaluation data models.

Defines typed structures for benchmark cases, evaluation results,
and evaluation reports.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
import uuid
import time


class BenchmarkCategory(Enum):
    """Benchmark case categories."""
    VOCABULARY = "vocabulary"
    SENTENCE_COMPLETION = "sentence_completion"
    COMPREHENSION = "comprehension"
    QUESTION_ANSWERING = "question_answering"
    INSTRUCTION_FOLLOWING = "instruction_following"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    CLASSIFICATION = "classification"
    REASONING = "reasoning"
    CONVERSATION = "conversation"


class MetricType(Enum):
    """Types of evaluation metrics."""
    EXACT_MATCH = "exact_match"
    CONTAINS = "contains"
    ACCURACY = "accuracy"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    TOKEN_COUNT = "token_count"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    HUMAN = "human"


class FailureCategory(Enum):
    """Categories for failed benchmark cases."""
    LANGUAGE_ERROR = "language_error"
    FACTUAL_ERROR = "factual_error"
    INSTRUCTION_FAILURE = "instruction_failure"
    REASONING_FAILURE = "reasoning_failure"
    FORMATTING_FAILURE = "formatting_failure"
    CONTEXT_FAILURE = "context_failure"
    GENERATION_FAILURE = "generation_failure"
    PROVIDER_ERROR = "provider_error"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


@dataclass
class HumanEvalScores:
    """Structured human evaluation scores (1-5 scale)."""
    coherence: Optional[int] = None
    relevance: Optional[int] = None
    correctness: Optional[int] = None
    instruction_following: Optional[int] = None
    language_quality: Optional[int] = None

    def average(self) -> float:
        scores = [
            s for s in [
                self.coherence, self.relevance, self.correctness,
                self.instruction_following, self.language_quality
            ] if s is not None
        ]
        return sum(scores) / len(scores) if scores else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "coherence": self.coherence,
            "relevance": self.relevance,
            "correctness": self.correctness,
            "instruction_following": self.instruction_following,
            "language_quality": self.language_quality,
            "average": self.average(),
        }


@dataclass
class BenchmarkCase:
    """A single benchmark evaluation case."""
    id: str
    category: BenchmarkCategory
    language: str  # "hy" for Armenian, "en" for English
    prompt: str
    expected_answer: Optional[str] = None
    reference_answers: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    conversation_history: Optional[List[Dict[str, str]]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category.value,
            "language": self.language,
            "prompt": self.prompt,
            "expected_answer": self.expected_answer,
            "reference_answers": self.reference_answers,
            "metadata": self.metadata,
            "tags": self.tags,
            "conversation_history": self.conversation_history,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BenchmarkCase":
        return cls(
            id=d["id"],
            category=BenchmarkCategory(d["category"]),
            language=d["language"],
            prompt=d["prompt"],
            expected_answer=d.get("expected_answer"),
            reference_answers=d.get("reference_answers", []),
            metadata=d.get("metadata", {}),
            tags=d.get("tags", []),
            conversation_history=d.get("conversation_history"),
        )


@dataclass
class CaseResult:
    """Result of evaluating a single benchmark case."""
    case_id: str
    prompt: str
    actual_response: str
    expected_answer: Optional[str]
    metrics: Dict[str, float] = field(default_factory=dict)
    passed: Optional[bool] = None
    failure_category: Optional[FailureCategory] = None
    failure_reason: Optional[str] = None
    latency_ms: float = 0.0
    token_count_input: int = 0
    token_count_output: int = 0
    human_scores: Optional[HumanEvalScores] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "case_id": self.case_id,
            "prompt": self.prompt,
            "actual_response": self.actual_response,
            "expected_answer": self.expected_answer,
            "metrics": self.metrics,
            "passed": self.passed,
            "failure_category": self.failure_category.value if self.failure_category else None,
            "failure_reason": self.failure_reason,
            "latency_ms": self.latency_ms,
            "token_count_input": self.token_count_input,
            "token_count_output": self.token_count_output,
        }
        if self.human_scores:
            result["human_scores"] = self.human_scores.to_dict()
        return result


@dataclass
class EvalRunConfig:
    """Configuration for an evaluation run."""
    provider_id: str
    model_id: str
    benchmark_name: str
    benchmark_version: str = "1.0"
    generation_config: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[str] = None
    run_id: Optional[str] = None

    def __post_init__(self):
        if self.run_id is None:
            ts = time.strftime("%Y_%m_%d_%H%M%S")
            short_id = uuid.uuid4().hex[:6]
            self.run_id = f"run_{ts}_{short_id}"
        if self.timestamp is None:
            self.timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "benchmark_name": self.benchmark_name,
            "benchmark_version": self.benchmark_version,
            "generation_config": self.generation_config,
            "environment": self.environment,
        }


@dataclass
class CategorySummary:
    """Summary for a single benchmark category."""
    category: str
    total_cases: int
    passed: int
    failed: int
    error: int
    pass_rate: float
    avg_latency_ms: float
    avg_throughput_tps: float
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class EvalReport:
    """Complete evaluation report."""
    run_config: EvalRunConfig
    case_results: List[CaseResult] = field(default_factory=list)
    category_summaries: List[CategorySummary] = field(default_factory=list)
    total_cases: int = 0
    total_passed: int = 0
    total_failed: int = 0
    total_errors: int = 0
    overall_pass_rate: float = 0.0
    overall_avg_latency_ms: float = 0.0
    overall_avg_throughput_tps: float = 0.0
    failures: List[CaseResult] = field(default_factory=list)

    def compute_summary(self):
        """Compute summary statistics from case results."""
        self.total_cases = len(self.case_results)
        self.total_passed = sum(1 for r in self.case_results if r.passed is True)
        self.total_failed = sum(1 for r in self.case_results if r.passed is False)
        self.total_errors = sum(1 for r in self.case_results if r.passed is None)
        self.overall_pass_rate = (
            self.total_passed / self.total_cases if self.total_cases > 0 else 0.0
        )

        latencies = [r.latency_ms for r in self.case_results if r.latency_ms > 0]
        self.overall_avg_latency_ms = sum(latencies) / len(latencies) if latencies else 0.0

        throughputs = []
        for r in self.case_results:
            if r.latency_ms > 0 and r.token_count_output > 0:
                tps = r.token_count_output / (r.latency_ms / 1000.0)
                throughputs.append(tps)
        self.overall_avg_throughput_tps = (
            sum(throughputs) / len(throughputs) if throughputs else 0.0
        )

        self.failures = [
            r for r in self.case_results
            if r.passed is False or r.passed is None
        ]

        self._compute_category_summaries()

    def _compute_category_summaries(self):
        """Compute per-category summaries."""
        from collections import defaultdict
        by_cat: Dict[str, List[CaseResult]] = defaultdict(list)
        for r in self.case_results:
            cat = r.metrics.get("_category", "unknown")
            by_cat[cat].append(r)

        self.category_summaries = []
        for cat_name, results in sorted(by_cat.items()):
            passed = sum(1 for r in results if r.passed is True)
            failed = sum(1 for r in results if r.passed is False)
            error = sum(1 for r in results if r.passed is None)
            total = len(results)
            latencies = [r.latency_ms for r in results if r.latency_ms > 0]
            throughputs = []
            for r in results:
                if r.latency_ms > 0 and r.token_count_output > 0:
                    throughputs.append(r.token_count_output / (r.latency_ms / 1000.0))

            self.category_summaries.append(CategorySummary(
                category=cat_name,
                total_cases=total,
                passed=passed,
                failed=failed,
                error=error,
                pass_rate=passed / total if total > 0 else 0.0,
                avg_latency_ms=sum(latencies) / len(latencies) if latencies else 0.0,
                avg_throughput_tps=sum(throughputs) / len(throughputs) if throughputs else 0.0,
            ))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_config": self.run_config.to_dict(),
            "summary": {
                "total_cases": self.total_cases,
                "total_passed": self.total_passed,
                "total_failed": self.total_failed,
                "total_errors": self.total_errors,
                "overall_pass_rate": self.overall_pass_rate,
                "overall_avg_latency_ms": self.overall_avg_latency_ms,
                "overall_avg_throughput_tps": self.overall_avg_throughput_tps,
            },
            "categories": [
                {
                    "category": s.category,
                    "total_cases": s.total_cases,
                    "passed": s.passed,
                    "failed": s.failed,
                    "error": s.error,
                    "pass_rate": s.pass_rate,
                    "avg_latency_ms": s.avg_latency_ms,
                    "avg_throughput_tps": s.avg_throughput_tps,
                }
                for s in self.category_summaries
            ],
            "failures": [r.to_dict() for r in self.failures],
            "results": [r.to_dict() for r in self.case_results],
        }
