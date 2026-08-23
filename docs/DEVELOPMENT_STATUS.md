# Voxline AI — Development Status

## Current Version: v0.4

**Maturity Level: Level 2 — Working Prototype**

Architecture is ahead of model intelligence. All components functional but model produces incoherent output.

## Test Status

| Test Suite | Status | Count |
|-----------|--------|-------|
| pytest (test_core.py) | PASS | 16/16 |
| pytest (test_business_agent.py) | PASS | 2/2 |
| pytest (test_architecture.py) | PASS | 35/35 |
| pytest (total) | PASS | **57/57** |
| Smoke tests (baseline_smoke.py) | PASS | **14/14** |
| Total | **PASS** | **71/71** |

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
| ToolRegistry | Working | 4 tools: calculator, file_read, file_write, directory_list |
| Planner | Working | Plan creation, step tracking, progress |
| ReasoningEngine | Working | Goal analysis, plan creation, revision decisions |
| AutonomousAgent | Working | Execution loop, state management |
| BusinessAgent | Working | Business plans, knowledge storage/retrieval |
| LocalVoxlineProvider | Working | Health check, generate, stream |
| LocalTransformersProvider | Working | HuggingFace model wrapping |
| ProviderFactory | Partial | Only "local" provider implemented |
| VoxlineConfig | Working | Environment-driven, defaults functional |
| ModelConfig | Working | Checkpoint compatibility, from_dict robust |
| Error hierarchy | Working | Centralized in src/errors.py |
| FastAPI Server | Working | /health, /chat, /generate, /docs |
| CLI chat.py | Working | Interactive chat with memory |

## Bugs Fixed (Phase 0 + Phase 1)

### Phase 0
1. **`src/agent/agent.py`**: Added missing `MemoryStore` import
2. **`src/providers/local_voxline.py:194`**: Fixed tensor dimension mismatch in `stream()`
3. **`src/config/model_config.py`**: Made `ModelConfig.from_dict()` robust to unknown/missing keys

### Phase 1
4. **`src/config/__init__.py`**: Fixed imports from absolute to relative (`from .model_config import ...`)
5. **`src/providers/__init__.py`**: Fixed imports from absolute to relative, fixed duplicate `__all__` override
6. **`src/checkpoint.py`**: Replaced local `CheckpointIncompatibilityError` with canonical import from `src/errors.py`
7. **`src/providers/factory.py`**: Replaced local `ProviderNotFoundError` with canonical import from `src/errors.py`

## Architecture Changes (Phase 1)

1. **Legacy modules moved**: `src/model.py`, `src/tokenizer.py`, `src/config.py`, `src/dataset.py`, `src/train.py`, `src/chat.py` moved to `src/legacy/`
2. **Error hierarchy created**: `src/errors.py` with 20+ exception classes inheriting from `VoxlineError`
3. **Empty directories removed**: `src/orchestrator/`, `src/coding/` (had no code)
4. **Placeholder package created**: `src/evaluation/` (ready for Phase 3)
5. **Type hints added**: Return type annotations on public methods across memory, planner, agent, tools, api modules
6. **Architecture tests added**: 35 tests covering canonical imports, legacy compatibility, model/tokenizer/memory/tools/config/planner architecture

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
| Providers | `src/providers/base.py` + implementations | Canonical |
| API | `src/api/chat.py`, `serve_v04.py` | Canonical |
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

## Performance Baseline

| Metric | Value |
|--------|-------|
| Model load time | ~0.93s |
| Inference latency | ~0.52s avg (50 tokens) |
| Throughput | ~96 tok/s (CPU) |
| Best model val_ppl | 135.78 |
| Best model val_loss | 4.911 |
| Training epochs | 9 (of 15 max) |
| Training time | ~32 min (CPU) |

## Known Limitations

1. **Model quality**: 936K params is too small for coherent generation. Perplexity 135.8.
2. **Training data**: ~4,628 of 5,064 Armenian lines are template-generated, not natural text.
3. **No GPU**: CPU-only limits model size and training speed.
4. **Two GenerationConfig classes**: `src.inference.generator.GenerationConfig` (max_new_tokens) vs `src.providers.base.GenerationConfig` (max_tokens). Kept separate — they serve different abstraction levels.
5. **Stub methods**: `Planner.decompose_task()` returns `[]`.
6. **No authentication**: API endpoints unauthenticated.
7. **No streaming**: FastAPI /chat doesn't support streaming responses.

## Git Status

- Branch: `main`
- 2 commits: `471abab feat: Voxline AI Core v0.3 stable`, `8d14de2 chore: establish v0.4 baseline`
- .gitignore present
- All v0.4 work committed
