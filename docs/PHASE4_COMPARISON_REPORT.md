# Phase 4 — Model Runtime Stabilization and Baseline Comparison

**Date:** 2026-08-24
**Status:** COMPLETE

## Summary

Phase 4 diagnosed and fixed the Qwen provider runtime failure caused by transformers 5.15.0 API changes, verified model integrity, and ran baseline evaluations comparing Native Voxline vs Qwen2.5-0.5B-Instruct across 37 benchmark cases (18 English + 19 Armenian).

## Root Cause Analysis

### The Problem
`QwenProvider.generate()` crashed with `KeyError: 'input_ids'` because:

1. **`apply_chat_template()` API change**: In transformers 5.15.0, `apply_chat_template(return_tensors="pt")` returns a `BatchEncoding` object (dict-like) instead of a plain tensor. The old code passed this directly to `model.generate()`, which tried to access `.shape[0]` on the dict, causing `KeyError`.

2. **`torch_dtype` deprecation**: transformers 5.15.0 renamed the `torch_dtype` parameter to `dtype` in `from_pretrained()`.

### The Fix
Applied in `src/providers/qwen_provider.py`:

```python
# Before (broken):
chat_output = self.tokenizer.apply_chat_template(messages, ..., return_tensors="pt")
# chat_output is BatchEncoding, not a tensor — model.generate() crashes

# After (fixed):
if hasattr(chat_output, "input_ids"):
    input_ids = chat_output.input_ids.to(self.device)
else:
    input_ids = chat_output.to(self.device)
```

```python
# Before (deprecated):
AutoModelForCausalLM.from_pretrained(..., torch_dtype=torch.float32)

# After (current API):
AutoModelForCausalLM.from_pretrained(..., dtype=torch.float32)
```

### Test Stability Fix
Tests created a new `QwenProvider` instance per test (each loading 942MB into 6GB RAM). Fixed by sharing a single instance via `setUpClass()`.

## Diagnostic Results

`scripts/diagnose_qwen_runtime.py` — 16 stages, all passed:

| Stage | Status | Detail |
|-------|--------|--------|
| A_python | OK | 3.13.14 |
| B_os | OK | Windows 10 |
| D_ram | OK | 5.9 GB total, 3.2 GB available |
| E_torch | OK | 2.13.0+cpu |
| F_transformers | OK | 5.15.0 |
| J_model_files | OK | All present, weights intact |
| K_config | OK | qwen2, vocab 151936, 32768 context |
| L_tokenizer_load | OK | vocab 151936, chat template present |
| M_model_load | OK | 494M params, float32, cpu |
| N_forward_pass | OK | logits shape [1, 17, 151936] |
| O_generate | OK | Generated coherent text |
| P_chat_template_generate | OK | Chat template + generate working |

## Baseline Evaluation Results

### English Benchmark (18 cases)

| Metric | Native Voxline | Qwen2.5-0.5B | Delta |
|--------|---------------|--------------|-------|
| **Pass Rate** | **0.0%** | **16.7%** | **+16.7%** |
| Avg Latency | 1,403 ms | 13,890 ms | +12,487 ms |
| Avg Throughput | 0.73 tok/s | 1.33 tok/s | +0.60 tok/s |
| Errors | 0 | 0 | — |

#### By Category (English)

| Category | Native Voxline | Qwen2.5-0.5B |
|----------|---------------|--------------|
| Vocabulary | 0/5 (0%) | 2/5 (40%) |
| Question Answering | 0/3 (0%) | 1/3 (33%) |
| Sentence Completion | 0/3 (0%) | 0/3 (0%) |
| Classification | 0/1 (0%) | 0/1 (0%) |
| Instruction Following | 0/2 (0%) | 0/2 (0%) |
| Reasoning | 0/2 (0%) | 0/2 (0%) |
| Translation | 0/2 (0%) | 0/2 (0%) |

### Armenian Benchmark (19 cases)

| Metric | Native Voxline | Qwen2.5-0.5B | Delta |
|--------|---------------|--------------|-------|
| **Pass Rate** | **0.0%** | **0.0%** | **0.0%** |
| Avg Latency | 1,226 ms | 18,262 ms | +17,036 ms |
| Avg Throughput | 0.83 tok/s | 0.67 tok/s | -0.16 tok/s |
| Errors | 0 | 0 | — |

### Key Observations

1. **Native Voxline generates garbage**: All outputs are incoherent mixed Armenian/Latin tokens (e.g., `մետpratմետիցվածonանcetէmaesինme.ինէին`). The 936K-param model is too small for meaningful generation.

2. **Qwen generates coherent English**: Correctly answers "What is the capital of France?" → "Paris", "Who wrote Romeo and Juliet?" → "William Shakespeare". Many "failures" are metrics being too strict (e.g., "The Earth revolves around the Sun" is correct but `sequence_similarity` with expected "Sun" is low).

3. **Qwen struggles with Armenian**: 0% pass rate on Armenian benchmarks. The model has limited Armenian capability — it generates plausible-looking Armenian text but doesn't answer the specific questions correctly.

4. **Latency tradeoff**: Qwen is ~10x slower than Native Voxline (13.9s vs 1.4s avg), but produces intelligible output. Native Voxline is fast but outputs garbage.

5. **No runtime errors**: Both providers completed all evaluations without crashes — the Qwen fix is stable.

## Files Modified

| File | Change |
|------|--------|
| `src/providers/qwen_provider.py` | Fixed BatchEncoding handling + dtype parameter |
| `tests/test_providers.py` | Shared Qwen provider instance via setUpClass |
| `scripts/diagnose_qwen_runtime.py` | New: 16-stage diagnostic script |
| `docs/PHASE4_COMPARISON_REPORT.md` | New: this report |

## Test Status

| Suite | Before Phase 4 | After Phase 4 |
|-------|----------------|---------------|
| pytest (non-Qwen) | 160/160 PASS | 160/160 PASS |
| pytest (Qwen) | 0/7 FAIL (KeyError) | 7/7 PASS |
| **Total** | **160/167** | **167/167** |
| Smoke tests | 14/14 PASS | 14/14 PASS |

## Evaluation Results Storage

```
eval_results/
  native_voxline_baseline/     — English: 18 cases, 0% pass
  native_voxline_armenian/     — Armenian: 19 cases, 0% pass
  qwen_baseline/               — English: 18 cases, 16.7% pass
  qwen_armenian_baseline/      — Armenian: 19 cases, 0% pass
  qwen_diagnostic.json         — 16-stage diagnostic results
```

## Conclusions

1. **Qwen provider is now stable** — runtime failure diagnosed, fixed, and verified with 7/7 tests passing.
2. **Qwen2.5-0.5B is viable for English tasks** — 16.7% pass rate on strict metrics, significantly better than Native Voxline's 0%.
3. **Armenian capability is limited** — both models score 0% on Armenian benchmarks, though Qwen generates plausible Armenian text.
4. **Evaluation framework works end-to-end** — all 4 evaluation runs completed, comparison detected regressions correctly.
5. **Infrastructure is ready** for Phase 5+ with a working multi-provider evaluation pipeline.
