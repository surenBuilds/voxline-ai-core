# Voxline AI — Development Status

## Current Version: v1.0.0

**Maturity Level: Level 2 — Working Prototype (v1.0 release)**

Architecture is ahead of model intelligence. All components functional but model produces incoherent output.

**v1.0.0 status:** Coding agent feature-complete for a v1 release after Phase 7 (Steps 1-14). All automated tests pass including real end-to-end validation against a local git repository (no hosted credentials or network required). See `docs/V1_RELEASE_REPORT.md`.

## Test Status

| Test Suite | Status | Count |
|-----------|--------|-------|
| pytest (test_core.py) | PASS | 20/20 |
| pytest (test_business_agent.py) | PASS | 2/2 |
| pytest (test_architecture.py) | PASS | 35/35 |
| pytest (test_providers.py) | PASS | 27/27 |
| pytest (test_evaluation.py) | PASS | 127/127 |
| pytest (test_assistant.py) | PASS | 28/28 |
| pytest (test_assistant_context.py) | PASS | 36/36 |
| pytest (test_assistant_chat.py) | PASS | 26/26 |
| pytest (test_assistant_business.py) | PASS | 55/55 |
| pytest (test_tools_security.py) | PASS | 73/73 (2 skipped: Windows symlink) |
| pytest (test_server.py) | PASS | 18/18 |
| pytest (test_language.py) | PASS | 29/29 |
| pytest (test_coding_agent.py) | PASS | 33/33 |
| pytest (test_github_integration.py) | PASS | 40/40 |
| pytest (test_vercel_integration.py) | PASS | 25/25 |
| pytest (test_tool_registration.py) | PASS | 32/32 |
| pytest (test_production_hardening.py) | PASS | 34/34 (includes Step 13 regressions) |
| pytest (e2e_coding_workflow.py) | PASS | 10/10 |
| pytest (smoke_real_integrations.py) | SKIP | 4/4 (gated behind VOXLINE_EXTERNAL_SMOKE=1) |
| pytest (test_armenian_benchmark.py) | PASS | 51/51 (15 skipped: live model) |
| pytest (test_real_e2e_validation.py) | PASS | 5/5 (gated behind VOXLINE_REAL_E2E=1) |
| Smoke tests (baseline_smoke.py) | PASS | 14/14 |
| Total (VOXLINE_REAL_E2E=1) | **PASS** | **677/677 (21 skipped)** |

## Component Status

| Component | Status | Notes |
|-----------|--------|-------|
| BPE Tokenizer | Working | 1109 vocab, encode/decode/save/load verified |
| VoxlineTransformer | Working | 936K params, forward/generate verified |
| Attention | Working | Causal + multi-head + scaled dot-product |
| Checkpoint save/load | Working | Roundtrip verified, config compatibility checking |
| TextGenerator | Working | Temperature, top-k, top-p, repetition penalty |
| ConversationalAI | Working | Multi-turn, memory integration, response cleaning |
| MemoryStore | Working | SQLite, search, CRUD, conversation memory |
| ToolRegistry | Working | 5 tools: calculator, file_read, file_write, directory_list, execute_command (Phase 7 Step 7: three-phase API, security integration) |
| Planner | Working | Plan creation, step tracking, progress |
| ReasoningEngine | Working | Goal analysis, plan creation, revision decisions |
| AutonomousAgent | Working | Execution loop, state management |
| BusinessAgent | Working | Business plans, knowledge storage/retrieval |
| **AIProvider ABC** | **Working** | chat, stream, health_check, generate, get_model_info |
| **LocalVoxlineProvider** | **Working** | Native Voxline, streaming, ModelInfo |
| **QwenProvider** | **Working** | HuggingFace Qwen2.5 local, chat template (Phase 4 fixed: BatchEncoding + dtype) |
| **ProviderFactory** | **Working** | Lazy registration, configurable creation (Phase 6: default=qwen) |
| VoxlineConfig | Working | Environment-driven, defaults functional (Phase 6: default provider=qwen) |
| ModelConfig | Working | Checkpoint compatibility, from_dict robust |
| Error hierarchy | Working | Centralized in src/errors.py |
| **PathSecurity** | **Working** | `Path.resolve()` + `is_relative_to()` workspace boundary (Phase 7 Step 7) |
| **CommandPolicy** | **Working** | Allowed/denied/approval command lists, blocked argument patterns (Phase 7 Step 7) |
| **CommandValidator** | **Working** | shlex parsing, cwd validation, subprocess(shell=False), timeout, output limit (Phase 7 Step 7) |
| **AuditLog** | **Working** | Append-only audit for all tool invocations, session tracking, no secrets (Phase 7 Step 7) |
| **FileSizeGuard** | **Working** | File size enforcement for reads and writes (Phase 7 Step 7) |
| **Tool Bootstrap** | **Working** | `build_tool_registry()` — conditional registration of core, GitHub, Vercel, workspace tools based on config/credentials. (Phase 7 Step 11) |
| **Capability Discovery** | **Working** | `ToolRegistry.available_tools()` — categorized tool summaries safe for LLM context. (Phase 7 Step 11) |
| **CodingAgent** | **Working** | Autonomous coding agent: plan → execute → validate → fix loop. Phase 11: repository context, 8-phase workflow (A-H), integration tool routing. Phase 12: CodingStatus, FailureType, operation_id, branch sanitization, deployment verification. Phase 13: workspace re-rooting preserves integration tools, feature branch auto-creation, fixed repo path double-nesting, AuditLog.log_event(). (Phase 7 Steps 8+11+12+13) |
| **GitHub Integration** | **Working** | Client, service, permission policy (READ/WRITE/DESTRUCTIVE), RepositoryWorkspace, tools. (Phase 7 Step 10) |
| **Vercel Integration** | **Working** | Client, service, permission policy (PREVIEW/PRODUCTION), deployment tools. (Phase 7 Step 10) |
| **CredentialProvider** | **Working** | Environment-based credential management with redaction. Tokens never exposed to LLM. (Phase 7 Step 10) |
| **Config Validation** | **Working** | `VoxlineConfig.validate()` — validates token presence, repo format, timeouts. (Phase 7 Step 12) |
| **GitHub Workflow** | **Working** | Commit/push separate from PR creation, branch sanitization, feature branch auto-created. (Phase 7 Step 12) |
| **Deployment Verification** | **Working** | `_verify_deployment()` polls Vercel API with timeout. (Phase 7 Step 12) |
| **FastAPI Server** | **Working** | Provider-configurable via --provider flag (Phase 6: default=qwen) |
| CLI chat.py | Working | Interactive chat with memory |
| **Evaluation Schemas** | **Working** | BenchmarkCase, CaseResult, EvalReport, HumanEvalScores, EvaluationStatus |
| **Evaluation Metrics** | **Working** | 20+ metric functions: exact, contains, smart_contains, task-specific (Phase 6) |
| **Evaluation Datasets** | **Working** | JSONL benchmark loading, filtering, built-in benchmarks |
| **Evaluation Runner** | **Working** | Provider evaluation orchestration, failure classification, task-specific pass logic |
| **Evaluation Reports** | **Working** | Text formatting, JSON save/load |
| **Evaluation Comparison** | **Working** | Run comparison, regression detection |
| **Text Normalization** | **Working** | Unicode, whitespace, numbers, punctuation, Armenian-specific (Phase 6 new) |
| **ChatAssistant** | **Working** | AIProvider integration, session management, context construction, selective memory (Phase 7) |
| **ContextBuilder** | **Working** | Memory injection, history budgeting, mode-specific context, ordered assembly (Phase 7) |
| **Session / SessionManager** | **Working** | In-memory session store, mode isolation, eviction, CRUD (Phase 7) |
| **BusinessAssistant** | **Working** | Business intelligence: 12 task types, structured context, KPI analysis, action planning (Phase 7) |
| **BusinessContext/Request/Response** | **Working** | Typed business models: context injection, request validation, structured responses (Phase 7) |
| **BusinessPlan/KPI/ActionItem/Recommendation** | **Working** | Structured planning: priorities, dependencies, risks, success metrics (Phase 7) |

## Bugs Fixed (Phase 0 + Phase 1 + Phase 2 + Phase 4)

### Phase 0
1. **`src/agent/agent.py`**: Added missing `MemoryStore` import
2. **`src/providers/local_voxline.py:194`**: Fixed tensor dimension mismatch in `stream()`
3. **`src/config/model_config.py`**: Made `ModelConfig.from_dict()` robust to unknown/missing keys

### Phase 1
4. **`src/config/__init__.py`**: Fixed imports from absolute to relative (`from .model_config import ...`)
5. **`src/providers/__init__.py`**: Fixed imports from absolute to relative, fixed duplicate `__all__` override
6. **`src/checkpoint.py`**: Replaced local `CheckpointIncompatibilityError` with canonical import from `src/errors.py`
7. **`src/providers/factory.py`**: Replaced local `ProviderNotFoundError` with canonical import from `src/errors.py`

### Phase 2
8. **`src/providers/base.py`**: Redesigned ABC with `chat()`, `get_model_info()`, `ModelInfo` dataclass, streaming defaults
9. **`src/providers/local_voxline.py`**: Updated to implement new ABC interface (added `model_id`, `supports_streaming`, `get_model_info()`, `chat()`)
10. **`src/providers/factory.py`**: Refactored to use lazy provider registration and configurable creation from VoxlineConfig
11. **`tests/test_architecture.py`**: Fixed import of `QwenProvider` (replaced removed `LocalTransformersProvider`)

### Phase 4
12. **`src/providers/qwen_provider.py`**: Fixed `apply_chat_template()` return type — extracts `.input_ids` from `BatchEncoding` (transformers 5.15.0 API change)
13. **`src/providers/qwen_provider.py`**: Replaced deprecated `torch_dtype` parameter with `dtype` (transformers 5.15.0 rename)
14. **`tests/test_providers.py`**: Changed `TestQwenProvider` to share single provider instance via `setUpClass` (memory constraint: 6GB RAM, 942MB model)

### Phase 7 Step 13 (v1.0 release — real E2E defects)
15. **`src/tools/tools.py`**: Added `ToolRegistry.set_workspace_root()` — re-roots the filesystem/command security boundary to a cloned repo WITHOUT dropping the already-registered GitHub/Vercel/workspace tools. Previously `_phase_workspace` replaced the registry with a fresh empty one, silently losing all integration tools so PR creation and deployment phases failed.
16. **`src/assistant/coding.py`**: Added `workspace_root` — `_phase_commit_and_push` / `_phase_review` / `_phase_vercel` now construct `RepositoryWorkspace` from the original workspace ROOT instead of the already-nested repo dir, fixing `NotADirectoryError` (double nesting) in the GitHub workflow.
17. **`src/assistant/coding.py`**: `_phase_workspace` now creates + checks out the feature branch after clone (previously all changes were made directly on the base branch and the feature-branch push failed with "refspec does not match any").
18. **`src/tools/security.py`**: Added `AuditLog.log_event()` — `RepositoryWorkspace._run_git()`/`clone()` called this method which did not exist, crashing when an audit log was supplied. Now records generic framework events without raising.

## Architecture Changes

### Phase 1
1. **Legacy modules moved**: `src/model.py`, `src/tokenizer.py`, `src/config.py`, `src/dataset.py`, `src/train.py`, `src/chat.py` moved to `src/legacy/`
2. **Error hierarchy created**: `src/errors.py` with 20+ exception classes inheriting from `VoxlineError`
3. **Empty directories removed**: `src/orchestrator/`, `src/coding/` (had no code)
4. **Placeholder package created**: `src/evaluation/` (ready for Phase 3)
5. **Type hints added**: Return type annotations on public methods across memory, planner, agent, tools, api modules
6. **Architecture tests added**: 35 tests covering canonical imports, legacy compatibility, model/tokenizer/memory/tools/config/planner architecture

### Phase 2
7. **AIProvider ABC redesigned**: Default `chat()` implementation, `get_model_info()` returning `ModelInfo` dataclass, `supports_streaming` property
8. **QwenProvider created**: `src/providers/qwen_provider.py` — wraps local Qwen2.5-0.5B-Instruct with HuggingFace transformers, uses `apply_chat_template`
9. **ProviderFactory refactored**: Lazy registration of built-in providers, creation driven by `VoxlineConfig.ai_provider`
10. **serve_v04.py made configurable**: `--provider` flag (native/qwen), uses `ProviderFactory.create()`
11. **Provider tests added**: 27 tests in `tests/test_providers.py` covering ABC contract, LocalVoxlineProvider, QwenProvider, ProviderFactory, interface compliance

### Phase 3
12. **Evaluation schemas created**: `src/evaluation/schemas.py` — BenchmarkCase, CaseResult, EvalRunConfig, EvalReport, CategorySummary, HumanEvalScores, FailureCategory, MetricType enums
13. **Evaluation metrics created**: `src/evaluation/metrics.py` — 12 metric functions covering exact match, contains, keyword, sequence similarity, word overlap, length ratio, number match, format check, context retention, classification accuracy
14. **Evaluation datasets created**: `src/evaluation/datasets.py` — JSONL benchmark loading, saving, built-in benchmark discovery, category/language/tag filtering
15. **Evaluation runner created**: `src/evaluation/runner.py` — EvaluationRunner with provider evaluation, timeout handling, failure classification, pass/fail determination
16. **Evaluation reports created**: `src/evaluation/reports.py` — Text report formatting, JSON save/load for results, summary, config
17. **Evaluation comparison created**: `src/evaluation/comparison.py` — Compare two runs, detect regressions by metric delta, save comparison to JSON
18. **Benchmarks created**: `benchmarks/armenian.jsonl` (19 cases), `benchmarks/english.jsonl` (18 cases) covering vocabulary, sentence completion, QA, instruction following, translation, reasoning, classification
19. **CLI entrypoints created**: `evaluate.py` (run evaluations), `compare_evaluations.py` (compare two runs)
20. **Evaluation tests added**: 83 tests in `tests/test_evaluation.py` covering schemas, metrics, datasets, runner, reports, comparison

## Canonical Module Map

| Module | Canonical Location | Status |
|--------|-------------------|--------|
| Model | `src/model/transformer.py` | Canonical |
| Tokenizer | `src/tokenizer/bpe.py` | Canonical |
| Training | `src/training/trainer.py` | Canonical |
| Inference | `src/inference/generator.py` | Canonical |
| Config | `src/config/model_config.py`, `src/config/settings.py` | Canonical |
| Memory | `src/memory/memory.py` | Canonical |
| Tools | `src/tools/tools.py` | Canonical |
| Planner | `src/planner/reasoning.py` | Canonical |
| Agent | `src/agent/agent.py` | Canonical |
| Business | `src/business/agent.py` | Canonical |
| **Providers** | `src/providers/base.py` + implementations | **Canonical (Phase 2 updated)** |
| API | `src/api/chat.py`, `serve_v04.py` | Canonical |
| **Evaluation** | `src/evaluation/` (schemas, metrics, datasets, runner, reports, comparison, normalize) | **Canonical (Phase 3 + Phase 6)** |
| Errors | `src/errors.py` | Canonical |
| Logging | `src/logging.py` | Canonical |
| Checkpoint | `src/checkpoint.py` | Canonical |

## Deprecated/Legacy Modules

| Module | Location | Status | Replacement |
|--------|----------|--------|-------------|
| VoxlineModel | `src/legacy/model.py` | Legacy | `src/model/transformer.py` |
| SimpleTokenizer | `src/legacy/tokenizer.py` | Legacy | `src/tokenizer/bpe.py` |
| TrainingConfig (HF) | `src/legacy/config.py` | Legacy | `src/training/trainer.py` |
| TextDataset | `src/legacy/dataset.py` | Legacy | `src/training/trainer.py` |
| train_model() | `src/legacy/train.py` | Legacy | `src/training/trainer.py` |
| ChatBot | `src/legacy/chat.py` | Legacy | `src/api/chat.py` |
| main.py | `main.py` (root) | Legacy | `scripts/train_small.py` |
| **LocalTransformersProvider** | `src/providers/local_transformers.py` | **Legacy** | `src/providers/qwen_provider.py` |

## Performance Baseline

| Metric | Native Voxline | Qwen2.5-0.5B |
|--------|---------------|--------------|
| Model load time | ~0.93s | ~3s |
| Inference latency (50 tok) | ~1.4s avg | ~13.9s avg |
| Throughput | ~0.73 tok/s | ~1.33 tok/s |
| English pass rate | 0.0% (0/18) | 16.7% (3/18) |
| Armenian pass rate | 0.0% (0/19) | 0.0% (0/19) |
| Best model val_ppl | 135.78 | N/A |
| Training epochs | 9 (of 15 max) | N/A |
| Training time | ~32 min (CPU) | N/A |

## Intelligence Assessment (Phase 5)

| Metric | Native Voxline | Qwen2.5-0.5B |
|--------|---------------|--------------|
| English true capability | 0% (incoherent output) | ~55-61% (metric-adjusted) |
| Armenian true capability | 0% (incoherent output) | ~0% (no semantic understanding) |
| Failure mode | Random character sequences | Verbose answers, language confusion |
| Viability | Not viable (retired as production target) | Viable for English tasks |

**Key insight:** 8 of 15 Qwen English "failures" are metric false negatives — the model answered correctly but strict scoring rejected it. See `docs/INTELLIGENCE_STRATEGY.md` for full analysis.

## Known Limitations

1. **Model quality**: Native Voxline (936K params) produces only incoherent mixed Armenian/Latin tokens. Retired as production target.
2. **Training data**: ~4,628 of 5,064 Armenian lines are template-generated, not natural text.
3. **No GPU**: CPU-only limits model size and training speed.
4. **Two GenerationConfig classes**: `src.inference.generator.GenerationConfig` (max_new_tokens) vs `src.providers.base.GenerationConfig` (max_tokens). Kept separate — they serve different abstraction levels.
5. **Stub methods**: `Planner.decompose_task()` returns `[]`.
6. **No authentication**: API endpoints unauthenticated.
7. **No streaming**: FastAPI /chat doesn't support streaming responses yet.
8. **Qwen Armenian capability**: Qwen2.5-0.5B scores 0% on Armenian benchmarks. Limited Armenian vocabulary and instruction following.
9. **Armenian benchmark quality**: Prompts contain typos, non-standard orthography, and template-generated text. Benchmark rewrite needed.
10. **Memory constraints**: 3.2 GB available RAM limits fine-tuning options to QLoRA only.
11. **Evaluation metrics**: Phase 6 added normalization and task-specific metrics, but V1 results remain unchanged for historical comparison.

## Git History

| Commit | Hash | Description |
|--------|------|-------------|
| feat: Voxline AI Core v0.3 stable | `471abab` | v0.3 release |
| chore: establish v0.4 baseline | `8d14de2` | Phase 0: safety + baseline |
| refactor: establish canonical core architecture | `88d0507` | Phase 1: clean architecture |
| feat: introduce unified AI provider architecture | `c477a7a` | Phase 2: provider system |
| feat: add Voxline evaluation framework | `3c0cc76` | Phase 3: evaluation system |
| fix: stabilize Qwen provider runtime | `a541afe` | Phase 4: runtime fix + baseline comparison |
| docs: intelligence strategy and model improvement roadmap | `833da85` | Phase 5: analysis, strategy, Phase 6 recommendation |
| feat: evaluation calibration + Qwen deployment foundation | `f1691a0` | Phase 6: normalization, task-specific metrics, default provider=qwen |
| feat: Phase 7 Steps 1-4 | `f1691a0` | Config extensions, error hierarchy, SessionManager, ContextBuilder, ChatAssistant |
| feat: Phase 7 Steps 5-6 | `6c95ebb` | BusinessAssistant with 12 task types |
| feat: Phase 7 Step 7 | `693db58` | Tool security hardening |
| feat: Phase 7 Step 8 | `e85e974` | CodingAgent autonomous coding |
| feat: Phase 7 Steps 9-10 | `2f9a3ca` | Integrations + GitHub + Vercel |
| feat: Phase 7 Step 11 | `7e13232` | Tool registration + E2E workflow |
| feat: Phase 7 Step 12 | TBD | Production hardening + security audit |
| feat: Phase 7 Steps 13-14 | TBD | Real E2E validation + production readiness (v1.0) |
| release: finalize coding agent v1.0.0 | TBD | v1.0.0 release commit |
