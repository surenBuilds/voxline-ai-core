"""
Evaluation framework tests.

Tests schemas, metrics, datasets, runner, reports, and comparison.
"""

import unittest
import asyncio
import json
import tempfile
from pathlib import Path

from src.evaluation.schemas import (
    BenchmarkCase, BenchmarkCategory, CaseResult,
    EvalRunConfig, EvalReport, CategorySummary,
    MetricType, FailureCategory, HumanEvalScores,
)
from src.evaluation.metrics import (
    exact_match, contains_match, keyword_match, sequence_similarity,
    word_overlap, length_ratio, number_match, format_check,
    context_retention_score, compute_case_metrics, aggregate_category_metrics,
    any_contains, best_reference_similarity, classification_accuracy,
)
from src.evaluation.datasets import (
    load_benchmark, save_benchmark, get_builtin_benchmarks,
    filter_cases,
)
from src.evaluation.runner import EvaluationRunner, EvaluationError
from src.evaluation.reports import (
    save_report, load_report, format_report_text, format_comparison_text,
)
from src.evaluation.comparison import (
    compare_reports, detect_regression, load_and_compare, save_comparison,
    ComparisonResult, MetricDelta,
)
from src.providers.base import AIProvider, GenerationConfig, ProviderHealth, ProviderStatus


# ---------------------------------------------------------------------------
# Mock provider for testing runner
# ---------------------------------------------------------------------------

class MockProvider(AIProvider):
    """Deterministic mock provider for evaluation tests."""

    def __init__(self, response_prefix="Response"):
        self._response_prefix = response_prefix
        self._call_count = 0

    @property
    def provider_id(self):
        return "mock_provider"

    @property
    def model_id(self):
        return "mock_model"

    @property
    def supports_streaming(self):
        return False

    async def generate(self, prompt, config):
        self._call_count += 1
        return f"{self._response_prefix} {self._call_count}: {prompt[:30]}"

    async def health_check(self):
        return ProviderHealth(
            status=ProviderStatus.HEALTHY,
            message="Mock provider ready",
            response_time_ms=1.0,
        )


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

class TestBenchmarkCase(unittest.TestCase):

    def test_create_case(self):
        case = BenchmarkCase(
            id="test_001",
            category=BenchmarkCategory.VOCABULARY,
            language="en",
            prompt="What is 2+2?",
            expected_answer="4",
        )
        self.assertEqual(case.id, "test_001")
        self.assertEqual(case.category, BenchmarkCategory.VOCABULARY)
        self.assertEqual(case.language, "en")
        self.assertEqual(case.expected_answer, "4")
        self.assertEqual(case.reference_answers, [])

    def test_to_dict(self):
        case = BenchmarkCase(
            id="test_001",
            category=BenchmarkCategory.REASONING,
            language="hy",
            prompt="test",
            expected_answer="answer",
            reference_answers=["alt"],
            tags=["math"],
        )
        d = case.to_dict()
        self.assertEqual(d["id"], "test_001")
        self.assertEqual(d["category"], "reasoning")
        self.assertEqual(d["language"], "hy")
        self.assertEqual(d["tags"], ["math"])

    def test_from_dict(self):
        d = {
            "id": "test_002",
            "category": "vocabulary",
            "language": "en",
            "prompt": "hello",
            "expected_answer": "world",
            "reference_answers": ["earth"],
            "metadata": {"pass_threshold": 0.3},
            "tags": ["basic"],
        }
        case = BenchmarkCase.from_dict(d)
        self.assertEqual(case.id, "test_002")
        self.assertEqual(case.category, BenchmarkCategory.VOCABULARY)
        self.assertEqual(case.metadata["pass_threshold"], 0.3)

    def test_from_dict_minimal(self):
        d = {
            "id": "test_003",
            "category": "comprehension",
            "language": "en",
            "prompt": "read this",
        }
        case = BenchmarkCase.from_dict(d)
        self.assertEqual(case.expected_answer, None)
        self.assertEqual(case.reference_answers, [])

    def test_roundtrip(self):
        case = BenchmarkCase(
            id="rt_001",
            category=BenchmarkCategory.TRANSLATION,
            language="hy",
            prompt="translate",
            expected_answer="done",
        )
        d = case.to_dict()
        case2 = BenchmarkCase.from_dict(d)
        self.assertEqual(case.id, case2.id)
        self.assertEqual(case.category, case2.category)
        self.assertEqual(case.language, case2.language)


class TestCaseResult(unittest.TestCase):

    def test_create_result(self):
        result = CaseResult(
            case_id="test_001",
            prompt="hello",
            actual_response="world",
            expected_answer="world",
            metrics={"exact_match": 1.0},
            passed=True,
            latency_ms=100.0,
        )
        self.assertTrue(result.passed)
        self.assertEqual(result.latency_ms, 100.0)

    def test_to_dict(self):
        result = CaseResult(
            case_id="r_001",
            prompt="p",
            actual_response="a",
            expected_answer="e",
            metrics={"em": 1.0},
            passed=True,
            latency_ms=50.0,
            token_count_input=5,
            token_count_output=10,
        )
        d = result.to_dict()
        self.assertEqual(d["case_id"], "r_001")
        self.assertEqual(d["latency_ms"], 50.0)
        self.assertEqual(d["token_count_output"], 10)

    def test_to_dict_with_human_scores(self):
        hs = HumanEvalScores(coherence=4, relevance=5, correctness=3)
        result = CaseResult(
            case_id="r_002", prompt="p", actual_response="a",
            expected_answer="e", human_scores=hs,
        )
        d = result.to_dict()
        self.assertIn("human_scores", d)
        self.assertEqual(d["human_scores"]["coherence"], 4)
        self.assertAlmostEqual(d["human_scores"]["average"], 4.0)


class TestEvalRunConfig(unittest.TestCase):

    def test_auto_generates_run_id(self):
        cfg = EvalRunConfig(
            provider_id="p",
            model_id="m",
            benchmark_name="b",
        )
        self.assertIsNotNone(cfg.run_id)
        self.assertTrue(cfg.run_id.startswith("run_"))
        self.assertIsNotNone(cfg.timestamp)

    def test_to_dict(self):
        cfg = EvalRunConfig(
            provider_id="p",
            model_id="m",
            benchmark_name="b",
            run_id="run_custom",
        )
        d = cfg.to_dict()
        self.assertEqual(d["run_id"], "run_custom")
        self.assertEqual(d["provider_id"], "p")


class TestEvalReport(unittest.TestCase):

    def test_compute_summary(self):
        report = EvalReport(
            run_config=EvalRunConfig(provider_id="p", model_id="m", benchmark_name="b"),
        )
        report.case_results = [
            CaseResult(case_id="1", prompt="p", actual_response="a", expected_answer="e",
                       passed=True, latency_ms=100, token_count_output=10, metrics={"_category": "vocabulary"}),
            CaseResult(case_id="2", prompt="p", actual_response="a", expected_answer="e",
                       passed=False, latency_ms=200, token_count_output=20, metrics={"_category": "vocabulary"}),
            CaseResult(case_id="3", prompt="p", actual_response="a", expected_answer="e",
                       passed=None, latency_ms=0, token_count_output=0, metrics={"_category": "reasoning"}),
        ]
        report.compute_summary()
        self.assertEqual(report.total_cases, 3)
        self.assertEqual(report.total_passed, 1)
        self.assertEqual(report.total_failed, 1)
        self.assertEqual(report.total_errors, 1)
        self.assertAlmostEqual(report.overall_pass_rate, 1 / 3)
        self.assertGreater(len(report.category_summaries), 0)

    def test_empty_report(self):
        report = EvalReport(
            run_config=EvalRunConfig(provider_id="p", model_id="m", benchmark_name="b"),
        )
        report.compute_summary()
        self.assertEqual(report.total_cases, 0)
        self.assertEqual(report.overall_pass_rate, 0.0)


class TestHumanEvalScores(unittest.TestCase):

    def test_average(self):
        hs = HumanEvalScores(coherence=4, relevance=5, correctness=3)
        self.assertAlmostEqual(hs.average(), 4.0)

    def test_average_partial(self):
        hs = HumanEvalScores(coherence=2)
        self.assertAlmostEqual(hs.average(), 2.0)

    def test_average_empty(self):
        hs = HumanEvalScores()
        self.assertAlmostEqual(hs.average(), 0.0)

    def test_to_dict(self):
        hs = HumanEvalScores(coherence=3, relevance=4, correctness=5)
        d = hs.to_dict()
        self.assertEqual(d["coherence"], 3)
        self.assertEqual(d["average"], 4.0)


# ---------------------------------------------------------------------------
# Metric tests
# ---------------------------------------------------------------------------

class TestExactMatch(unittest.TestCase):

    def test_match(self):
        self.assertEqual(exact_match("hello", "hello"), 1.0)

    def test_case_insensitive(self):
        self.assertEqual(exact_match("Hello", "hello"), 1.0)

    def test_mismatch(self):
        self.assertEqual(exact_match("hello", "world"), 0.0)

    def test_whitespace_stripped(self):
        self.assertEqual(exact_match("  hello  ", "hello"), 1.0)


class TestContainsMatch(unittest.TestCase):

    def test_contains(self):
        self.assertEqual(contains_match("hello world", "world"), 1.0)

    def test_not_contains(self):
        self.assertEqual(contains_match("hello", "world"), 0.0)

    def test_case_insensitive(self):
        self.assertEqual(contains_match("Hello World", "world"), 1.0)


class TestKeywordMatch(unittest.TestCase):

    def test_all_found(self):
        self.assertEqual(keyword_match("red blue green", ["red", "blue", "green"]), 1.0)

    def test_partial(self):
        self.assertAlmostEqual(keyword_match("red blue", ["red", "blue", "green"]), 2 / 3)

    def test_none_found(self):
        self.assertEqual(keyword_match("hello", ["red", "blue"]), 0.0)

    def test_empty_keywords(self):
        self.assertEqual(keyword_match("hello", []), 0.0)


class TestSequenceSimilarity(unittest.TestCase):

    def test_identical(self):
        self.assertAlmostEqual(sequence_similarity("hello", "hello"), 1.0)

    def test_empty(self):
        self.assertAlmostEqual(sequence_similarity("", ""), 1.0)

    def test_different(self):
        sim = sequence_similarity("hello", "world")
        self.assertLess(sim, 1.0)

    def test_partial(self):
        sim = sequence_similarity("hello world", "hello earth")
        self.assertGreater(sim, 0.3)
        self.assertLess(sim, 1.0)


class TestWordOverlap(unittest.TestCase):

    def test_identical(self):
        self.assertAlmostEqual(word_overlap("a b c", "a b c"), 1.0)

    def test_no_overlap(self):
        self.assertEqual(word_overlap("a b", "c d"), 0.0)

    def test_partial(self):
        overlap = word_overlap("a b c", "b c d")
        self.assertGreater(overlap, 0.0)
        self.assertLess(overlap, 1.0)


class TestLengthRatio(unittest.TestCase):

    def test_identical_length(self):
        self.assertAlmostEqual(length_ratio("abc", "xyz"), 1.0)

    def test_shorter(self):
        self.assertGreater(length_ratio("ab", "abcd"), 0.0)
        self.assertLess(length_ratio("ab", "abcd"), 1.0)

    def test_empty_expected(self):
        self.assertEqual(length_ratio("abc", ""), 0.0)


class TestNumberMatch(unittest.TestCase):

    def test_match(self):
        self.assertEqual(number_match("The answer is 42", "42"), 1.0)

    def test_no_match(self):
        self.assertEqual(number_match("The answer is 42", "43"), 0.0)

    def test_floating_point(self):
        self.assertEqual(number_match("3.14 pi", "3.14"), 1.0)


class TestFormatCheck(unittest.TestCase):

    def test_numbered_list_pass(self):
        text = "1. First item\n2. Second item\n3. Third item"
        self.assertEqual(format_check(text, "numbered_list"), 1.0)

    def test_numbered_list_fail(self):
        self.assertEqual(format_check("just text", "numbered_list"), 0.0)

    def test_short_answer_pass(self):
        self.assertEqual(format_check("short", "short_answer"), 1.0)

    def test_short_answer_fail(self):
        long_text = " ".join(["word"] * 25)
        self.assertEqual(format_check(long_text, "short_answer"), 0.0)

    def test_one_word_pass(self):
        self.assertEqual(format_check("yes", "one_word"), 1.0)

    def test_bullet_list_pass(self):
        text = "- item one\n- item two\n- item three"
        self.assertEqual(format_check(text, "bullet_list"), 1.0)

    def test_unknown_format(self):
        self.assertEqual(format_check("anything", "unknown"), 1.0)


class TestContextRetentionScore(unittest.TestCase):

    def test_retains_context(self):
        history = [{"role": "user", "content": "My name is Alice"}]
        self.assertEqual(
            context_retention_score(history, "Alice, nice to meet you", "Alice"),
            1.0,
        )

    def test_no_context(self):
        history = [{"role": "user", "content": "My name is Alice"}]
        self.assertEqual(
            context_retention_score(history, "Hello stranger", "Alice"),
            0.0,
        )


class TestAnyContains(unittest.TestCase):

    def test_match(self):
        self.assertEqual(any_contains("hello", ["world", "hello"]), 1.0)

    def test_no_match(self):
        self.assertEqual(any_contains("hi", ["world", "hello"]), 0.0)

    def test_empty_refs(self):
        self.assertEqual(any_contains("hello", []), 0.0)


class TestBestReferenceSimilarity(unittest.TestCase):

    def test_best_match(self):
        refs = ["hello world", "goodbye earth"]
        score = best_reference_similarity("hello world", refs)
        self.assertAlmostEqual(score, 1.0)

    def test_no_refs(self):
        self.assertEqual(best_reference_similarity("hello", []), 0.0)


class TestClassificationAccuracy(unittest.TestCase):

    def test_correct(self):
        self.assertEqual(
            classification_accuracy("positive", "positive", ["positive", "negative"]),
            1.0,
        )

    def test_incorrect(self):
        self.assertEqual(
            classification_accuracy("positive", "negative", ["positive", "negative"]),
            0.0,
        )

    def test_label_in_text(self):
        self.assertEqual(
            classification_accuracy("I think positive", "positive", ["positive", "negative"]),
            1.0,
        )


class TestComputeCaseMetrics(unittest.TestCase):

    def test_with_expected(self):
        metrics = compute_case_metrics(
            actual="Paris",
            expected="Paris",
            references=["Paris", "The capital of France is Paris"],
            category="vocabulary",
        )
        self.assertEqual(metrics["exact_match"], 1.0)
        self.assertEqual(metrics["contains"], 1.0)
        self.assertGreater(metrics["sequence_similarity"], 0.5)

    def test_with_references(self):
        metrics = compute_case_metrics(
            actual="The capital of France is Paris",
            expected=None,
            references=["Paris", "The capital of France is Paris"],
        )
        self.assertGreater(metrics.get("any_reference_match", 0), 0)
        self.assertGreater(metrics.get("best_reference_similarity", 0), 0.5)

    def test_no_expected(self):
        metrics = compute_case_metrics(actual="anything", expected=None)
        self.assertNotIn("exact_match", metrics)


class TestAggregateCategoryMetrics(unittest.TestCase):

    def test_aggregate(self):
        results = [
            {"exact_match": 1.0, "contains": 0.5},
            {"exact_match": 0.0, "contains": 1.0},
        ]
        agg = aggregate_category_metrics(results)
        self.assertAlmostEqual(agg["exact_match"], 0.5)
        self.assertAlmostEqual(agg["contains"], 0.75)

    def test_empty(self):
        self.assertEqual(aggregate_category_metrics([]), {})


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------

class TestDatasets(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.benchmark_file = Path(self.tmpdir) / "test_benchmark.jsonl"
        self.cases = [
            BenchmarkCase(
                id="d_001",
                category=BenchmarkCategory.VOCABULARY,
                language="en",
                prompt="What is 2+2?",
                expected_answer="4",
                tags=["basic"],
            ),
            BenchmarkCase(
                id="d_002",
                category=BenchmarkCategory.REASONING,
                language="hy",
                prompt="test reasoning",
                expected_answer="answer",
                tags=["math"],
            ),
        ]
        save_benchmark(self.cases, self.benchmark_file)

    def test_save_and_load(self):
        loaded = load_benchmark(self.benchmark_file)
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0].id, "d_001")
        self.assertEqual(loaded[1].language, "hy")

    def test_load_nonexistent(self):
        with self.assertRaises(FileNotFoundError):
            load_benchmark("/nonexistent/file.jsonl")

    def test_get_builtin_benchmarks(self):
        benchmarks = get_builtin_benchmarks()
        self.assertIsInstance(benchmarks, dict)
        self.assertIn("armenian", benchmarks)
        self.assertIn("english", benchmarks)

    def test_filter_by_language(self):
        loaded = load_benchmark(self.benchmark_file)
        hy = filter_cases(loaded, languages=["hy"])
        self.assertEqual(len(hy), 1)
        self.assertEqual(hy[0].language, "hy")

    def test_filter_by_category(self):
        loaded = load_benchmark(self.benchmark_file)
        vocab = filter_cases(loaded, categories=["vocabulary"])
        self.assertEqual(len(vocab), 1)

    def test_filter_by_tags(self):
        loaded = load_benchmark(self.benchmark_file)
        math = filter_cases(loaded, tags=["math"])
        self.assertEqual(len(math), 1)

    def test_filter_multiple_criteria(self):
        loaded = load_benchmark(self.benchmark_file)
        result = filter_cases(loaded, languages=["en"], categories=["vocabulary"])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].id, "d_001")


# ---------------------------------------------------------------------------
# Runner tests
# ---------------------------------------------------------------------------

class TestEvaluationRunner(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.benchmark_file = Path(self.tmpdir) / "runner_test.jsonl"
        self.cases = [
            BenchmarkCase(
                id="r_001",
                category=BenchmarkCategory.VOCABULARY,
                language="en",
                prompt="What is 2+2?",
                expected_answer="4",
                reference_answers=["4", "four"],
                metadata={"pass_threshold": 0.3},
            ),
            BenchmarkCase(
                id="r_002",
                category=BenchmarkCategory.REASONING,
                language="en",
                prompt="Capital of France?",
                expected_answer="Paris",
                reference_answers=["Paris"],
                metadata={"pass_threshold": 0.3},
            ),
        ]
        save_benchmark(self.cases, self.benchmark_file)
        self.provider = MockProvider(response_prefix="Answer")
        self.gen_config = GenerationConfig(max_tokens=20, temperature=0.7)

    def test_run_full(self):
        runner = EvaluationRunner(self.provider, self.gen_config)
        report = runner.run(self.benchmark_file)
        self.assertIsInstance(report, EvalReport)
        self.assertEqual(len(report.case_results), 2)
        self.assertEqual(report.total_cases, 2)

    def test_run_with_filters(self):
        runner = EvaluationRunner(self.provider, self.gen_config)
        report = runner.run(self.benchmark_file, categories=["vocabulary"])
        self.assertEqual(len(report.case_results), 1)

    def test_run_cases(self):
        runner = EvaluationRunner(self.provider, self.gen_config)
        report = runner.run_cases(self.cases)
        self.assertEqual(len(report.case_results), 2)
        self.assertGreater(report.overall_avg_latency_ms, 0)

    def test_provider_error_handling(self):
        class ErrorProvider(AIProvider):
            @property
            def provider_id(self): return "error"
            @property
            def model_id(self): return "error_model"
            @property
            def supports_streaming(self): return False
            async def generate(self, prompt, config):
                raise RuntimeError("Model crashed")
            async def health_check(self):
                return ProviderHealth(ProviderStatus.UNAVAILABLE, "down")

        runner = EvaluationRunner(ErrorProvider(), self.gen_config)
        report = runner.run(self.benchmark_file)
        self.assertEqual(report.total_errors, 2)
        for r in report.case_results:
            self.assertIsNone(r.passed)
            self.assertEqual(r.failure_category, FailureCategory.PROVIDER_ERROR)

    def test_metrics_populated(self):
        runner = EvaluationRunner(self.provider, self.gen_config)
        report = runner.run_cases(self.cases)
        for r in report.case_results:
            self.assertIn("exact_match", r.metrics)
            self.assertIn("_category", r.metrics)
            self.assertIn("_language", r.metrics)

    def test_failure_classification(self):
        class EmptyProvider(AIProvider):
            @property
            def provider_id(self): return "empty"
            @property
            def model_id(self): return "empty_model"
            @property
            def supports_streaming(self): return False
            async def generate(self, prompt, config):
                return ""
            async def health_check(self):
                return ProviderHealth(ProviderStatus.HEALTHY, "ok")

        runner = EvaluationRunner(EmptyProvider(), self.gen_config)
        report = runner.run_cases([
            BenchmarkCase(
                id="empty_001",
                category=BenchmarkCategory.VOCABULARY,
                language="en",
                prompt="test",
                expected_answer="expected",
            )
        ])
        self.assertEqual(report.total_failed, 1)
        self.assertIsNotNone(report.failures[0].failure_category)


# ---------------------------------------------------------------------------
# Report tests
# ---------------------------------------------------------------------------

class TestReports(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.report = EvalReport(
            run_config=EvalRunConfig(
                provider_id="test_provider",
                model_id="test_model",
                benchmark_name="test_bench",
                run_id="run_test_001",
            ),
        )
        self.report.case_results = [
            CaseResult(case_id="1", prompt="p", actual_response="a", expected_answer="e",
                       passed=True, latency_ms=100, token_count_output=10,
                       metrics={"exact_match": 1.0, "_category": "vocabulary", "_language": "en"}),
            CaseResult(case_id="2", prompt="p", actual_response="b", expected_answer="e",
                       passed=False, latency_ms=200, token_count_output=5,
                       metrics={"exact_match": 0.0, "_category": "reasoning", "_language": "hy"}),
        ]
        self.report.compute_summary()

    def test_save_and_load(self):
        output_dir = Path(self.tmpdir) / "report_test"
        save_report(self.report, output_dir)
        self.assertTrue((output_dir / "results.json").exists())
        self.assertTrue((output_dir / "summary.json").exists())
        self.assertTrue((output_dir / "config.json").exists())

        loaded = load_report(output_dir)
        self.assertEqual(len(loaded.case_results), 2)
        self.assertEqual(loaded.total_passed, 1)
        self.assertEqual(loaded.total_failed, 1)

    def test_format_report_text(self):
        text = format_report_text(self.report)
        self.assertIn("VOXLINE MODEL EVALUATION", text)
        self.assertIn("test_provider", text)
        self.assertIn("test_model", text)
        self.assertIn("50.0%", text)

    def test_format_comparison_text(self):
        report2 = EvalReport(
            run_config=EvalRunConfig(
                provider_id="provider_b",
                model_id="model_b",
                benchmark_name="test",
                run_id="run_002",
            ),
        )
        report2.case_results = [
            CaseResult(case_id="1", prompt="p", actual_response="a", expected_answer="e",
                       passed=True, latency_ms=150, token_count_output=10,
                       metrics={"_category": "vocabulary", "_language": "en"}),
        ]
        report2.compute_summary()

        text = format_comparison_text(self.report, report2, "A", "B")
        self.assertIn("EVALUATION COMPARISON", text)
        self.assertIn("A", text)
        self.assertIn("B", text)


# ---------------------------------------------------------------------------
# Comparison tests
# ---------------------------------------------------------------------------

class TestComparison(unittest.TestCase):

    def _make_report(self, pass_rate, latency, provider_id="p", run_id="r"):
        report = EvalReport(
            run_config=EvalRunConfig(
                provider_id=provider_id,
                model_id="m",
                benchmark_name="b",
                run_id=run_id,
            ),
        )
        total = 10
        passed_count = int(total * pass_rate)
        for i in range(total):
            report.case_results.append(
                CaseResult(
                    case_id=str(i),
                    prompt="p",
                    actual_response="a",
                    expected_answer="e",
                    passed=i < passed_count,
                    latency_ms=latency,
                    token_count_output=10,
                    metrics={"_category": "vocabulary", "_language": "en"},
                )
            )
        report.compute_summary()
        return report

    def test_compare_reports(self):
        a = self._make_report(0.5, 100, "provider_a", "run_a")
        b = self._make_report(0.8, 80, "provider_b", "run_b")
        result = compare_reports(a, b)
        self.assertIsInstance(result, ComparisonResult)
        self.assertGreater(len(result.deltas), 0)

    def test_detect_regression(self):
        a = self._make_report(0.9, 100)
        b = self._make_report(0.3, 200)
        regressions = detect_regression(a, b, threshold=0.1)
        self.assertGreater(len(regressions), 0)

    def test_no_regression(self):
        a = self._make_report(0.5, 100)
        b = self._make_report(0.6, 100)
        regressions = detect_regression(a, b, threshold=0.2)
        self.assertEqual(len(regressions), 0)

    def test_save_comparison(self):
        a = self._make_report(0.5, 100, "pa", "ra")
        b = self._make_report(0.7, 80, "pb", "rb")
        result = compare_reports(a, b)
        output = Path(tempfile.mkdtemp()) / "comparison.json"
        save_comparison(result, output)
        self.assertTrue(output.exists())
        with open(output) as f:
            data = json.load(f)
        self.assertEqual(data["provider_a"], "pa")
        self.assertEqual(data["provider_b"], "pb")

    def test_metric_delta(self):
        d = MetricDelta(
            metric_name="pass_rate",
            value_a=0.5,
            value_b=0.7,
            delta=0.2,
            direction="higher_is_better",
            regressed=False,
        )
        self.assertFalse(d.regressed)
        self.assertAlmostEqual(d.delta, 0.2)


if __name__ == "__main__":
    unittest.main()
