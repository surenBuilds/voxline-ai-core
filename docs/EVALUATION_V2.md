# Evaluation V2 — Normalization & Task-Specific Metrics

Phase 6 of the Voxline AI Core development plan.

## Overview

Evaluation V2 adds normalization, task-specific metrics, and a structured
evaluation status system to the existing evaluation framework.

## Changes from V1

### Text Normalization (`src/evaluation/normalize.py`)

All text comparisons go through a normalization pipeline:

| Step | Function | Effect |
|------|----------|--------|
| 1 | `normalize_unicode()` | NFKC standard form |
| 2 | `normalize_whitespace()` | Collapse multiple spaces, strip |
| 3 | `normalize_numbers()` | Remove commas in numbers |
| 4 | `normalize_punctuation_*()` | Normalize Armenian/English punctuation |
| 5 | `normalize_case()` | Lowercase for comparison |

Language-specific normalization handles Armenian quotation marks, hyphens,
and period/comma spacing.

### Smart Contains (`smart_contains`)

Replaces simple substring matching. Applies the full normalization pipeline
before checking containment. Prevents false negatives from case/whitespace.

### Task-Specific Metrics

Each benchmark category now has a primary metric:

| Category | Primary Metric | Logic |
|----------|---------------|-------|
| `vocabulary` | `vocabulary_accuracy` | Normalized contains check |
| `sentence_completion` | `sentence_completion_match` | Contains + similarity |
| `question_answering` | `qa_match` | Exact → normalized → number → contains → reference → similarity |
| `instruction_following` | `instruction_following_score` | Contains + similarity |
| `translation` | `translation_score` | Multi-reference with similarity threshold |
| `reasoning` | `reasoning_score` | Number match + normalized contains |
| `classification` | `classification_accuracy` | Label extraction + match |

### Evaluation Status (`EvaluationStatus`)

New enum: `PASS`, `PARTIAL`, `FAIL`, `INVALID_EVALUATION`

### Human Evaluation Notes

`HumanEvalScores` now includes a `notes` field for annotator observations.

## Running Evaluation

```bash
# English benchmark
python -m src.evaluation.runner benchmarks/english.jsonl

# Armenian benchmark
python -m src.evaluation.runner benchmarks/armenian.jsonl

# Run all benchmarks
python scripts/run_benchmark.py --all
```

## Benchmark Files

- `benchmarks/english.jsonl` — 18 English cases (v1, preserved)
- `benchmarks/armenian.jsonl` — 19 Armenian cases (v1, preserved)

Both v1 files are preserved unchanged. New v2 benchmarks will be added
in future phases if needed.

## Impact on Existing Results

V1 evaluation results (`eval_results/`) used the old metrics.
Re-evaluation with V2 metrics would change pass/fail determinations.
Results are preserved for historical comparison.
