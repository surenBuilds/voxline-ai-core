# Voxline AI — Evaluation Framework

## Overview

The evaluation framework provides benchmarking, metrics, comparison, and regression detection for Voxline AI models and providers.

## Components

```
src/evaluation/
  schemas.py       Data models: BenchmarkCase, CaseResult, EvalReport, etc.
  metrics.py       Metric functions: exact_match, contains, similarity, format checks
  datasets.py      Benchmark loading, saving, filtering from JSONL files
  runner.py        EvaluationRunner: orchestrates provider evaluation against benchmarks
  reports.py       Report generation: text formatting, save/load JSON
  comparison.py    Report comparison, regression detection
  __init__.py      Public API exports
```

## Benchmark Format (JSONL)

Each line is a JSON object:

```json
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
```

### Required Fields
- `id`: Unique case identifier
- `category`: One of `vocabulary`, `sentence_completion`, `comprehension`, `question_answering`, `instruction_following`, `translation`, `summarization`, `classification`, `reasoning`, `conversation`
- `language`: `"hy"` (Armenian) or `"en"` (English)
- `prompt`: Input text

### Optional Fields
- `expected_answer`: Single expected answer
- `reference_answers`: List of acceptable answers
- `metadata`: Dict with `pass_threshold` (0.0–1.0, default 0.5)
- `tags`: List of tags for filtering
- `conversation_history`: Multi-turn context

## Built-in Benchmarks

| File | Language | Cases | Categories |
|------|----------|-------|------------|
| `benchmarks/armenian.jsonl` | Armenian | 19 | vocabulary, sentence_completion, question_answering, instruction_following, translation, reasoning |
| `benchmarks/english.jsonl` | English | 18 | vocabulary, sentence_completion, question_answering, instruction_following, reasoning, translation, classification |

## Metrics

| Metric | Range | Description |
|--------|-------|-------------|
| `exact_match` | 0.0–1.0 | Case-insensitive exact string match |
| `contains` | 0.0–1.0 | Expected substring found in response |
| `sequence_similarity` | 0.0–1.0 | SequenceMatcher ratio |
| `word_overlap` | 0.0–1.0 | Jaccard word overlap |
| `keyword_match` | 0.0–1.0 | Fraction of keywords found |
| `number_match` | 0.0–1.0 | Numeric value match |
| `format_check` | 0.0–1.0 | Format compliance (numbered_list, bullet_list, short_answer, one_word) |
| `any_reference_match` | 0.0–1.0 | Response contains any reference answer |
| `best_reference_similarity` | 0.0–1.0 | Highest similarity across references |
| `classification_accuracy` | 0.0–1.0 | Label classification accuracy |
| `context_retention_score` | 0.0–1.0 | Conversation context retention |

## Usage

### Python API

```python
from src.evaluation import EvaluationRunner, load_benchmark
from src.providers.base import GenerationConfig

# Load provider
provider = ...  # any AIProvider instance

# Configure generation
gen_config = GenerationConfig(max_tokens=150, temperature=0.7)

# Run evaluation
runner = EvaluationRunner(provider, gen_config)
report = runner.run("benchmarks/english.jsonl")

# Filter by category/language
report = runner.run("benchmarks/armenian.jsonl", categories=["vocabulary"], languages=["hy"])

# Print report
from src.evaluation import format_report_text
print(format_report_text(report))

# Save results
from src.evaluation import save_report
save_report(report, "eval_results/")
```

### CLI

```bash
# List available benchmarks
python evaluate.py --list-benchmarks

# Run evaluation
python evaluate.py --provider qwen --benchmark benchmarks/english.jsonl
python evaluate.py --provider native --benchmark armenian --categories vocabulary --languages hy

# Save results
python evaluate.py --provider qwen --benchmark english --output eval_results/

# Compare two runs
python compare_evaluations.py eval_run_a/ eval_run_b/
python compare_evaluations.py eval_run_a/ eval_run_b/ --threshold 0.05
```

## Evaluation Runner

`EvaluationRunner` orchestrates evaluation:

1. Loads benchmark cases from JSONL
2. Optionally filters by category, language, tags
3. Sends each prompt to the provider
4. Computes metrics per case
5. Classifies failures (language_error, factual_error, reasoning_failure, etc.)
6. Aggregates results by category
7. Produces `EvalReport`

### Pass/Fail Logic

A case passes if any of:
- `exact_match` ≥ 1.0
- `number_match` ≥ 1.0
- `classification_accuracy` ≥ 1.0
- `keyword_match` ≥ 0.8
- Best similarity score ≥ `pass_threshold` (from metadata)

## Comparison

Compare two evaluation runs:

```python
from src.evaluation import compare_reports, detect_regression

result = compare_reports(report_a, report_b)

# Check for regressions
regressions = detect_regression(report_a, report_b, threshold=0.05)
for r in regressions:
    print(f"{r.metric_name}: {r.value_a:.3f} -> {r.value_b:.3f}")
```

Regression detection compares:
- Overall pass rate (higher is better)
- Average latency (lower is better)
- Average throughput (higher is better)
- Per-category pass rates

## Test Coverage

83 tests in `tests/test_evaluation.py`:
- Schema tests: BenchmarkCase, CaseResult, EvalRunConfig, EvalReport, HumanEvalScores
- Metric tests: 14 metric functions tested individually
- Dataset tests: save/load, filtering, built-in benchmarks
- Runner tests: full evaluation, filters, provider error handling, failure classification
- Report tests: save/load, text formatting
- Comparison tests: compare_reports, detect_regression, save/load
