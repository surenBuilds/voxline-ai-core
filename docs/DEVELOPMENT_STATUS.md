# Voxline AI — Development Status

## Current Version: v0.4

**Maturity Level: Level 2 — Working Prototype**

Architecture is ahead of model intelligence. All components functional but model produces incoherent output.

## Test Status

| Test Suite | Status | Count |
|-----------|--------|-------|
| pytest (test_core.py) | PASS | 16/16 |
| pytest (test_business_agent.py) | PASS | 2/2 |
| pytest (total) | PASS | 22/22 |
| Smoke tests (baseline_smoke.py) | PASS | 14/14 |
| Total | **PASS** | **36/36** |

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
| LocalVoxlineProvider | Working | Health check, generate, stream (minor tensor bug fixed) |
| LocalTransformersProvider | Working | HuggingFace model wrapping |
| ProviderFactory | Partial | Only "local" provider implemented |
| VoxlineConfig | Working | Environment-driven, defaults functional |
| ModelConfig | Working | Checkpoint compatibility, forward/backward compatible from_dict |
| FastAPI Server | Working | /health, /chat, /generate, /docs |
| CLI chat.py | Working | Interactive chat with memory |

## Bugs Fixed in This Baseline

1. **`src/agent/agent.py`**: Added missing `MemoryStore` import (was NameError at class definition time)
2. **`src/providers/local_voxline.py:194`**: Fixed tensor dimension mismatch in `stream()` — removed erroneous `.unsqueeze(0)` on `next_token` that produced 3D tensor
3. **`src/config/model_config.py`**: Made `ModelConfig.from_dict()` robust to unknown keys and missing required fields (forward/backward compatible)

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

## Corpus Stats

| Metric | Value |
|--------|-------|
| Total lines | 7,312 |
| Armenian | 5,064 (47.9%) |
| English | 2,248 (38.2%) |
| Unique words | 40,468 |
| Template-generated | ~4,628 (Armenian) |
| Real sentences | ~436 (Armenian) |

## File Inventory

### Core Source (src/)
```
src/agent/agent.py          AutonomousAgent, AgentState, ExecutionLog
src/api/chat.py             ConversationalAI
src/api/server.py           FastAPI endpoints (legacy)
src/attention/attention.py  ScaledDotProductAttention, MultiHeadAttention, CausalSelfAttention
src/business/agent.py       BusinessAgent, BusinessPlan
src/checkpoint.py           CheckpointLoader
src/config/model_config.py  ModelConfig, ModelType
src/config/settings.py      VoxlineConfig
src/inference/generator.py  TextGenerator, GenerationConfig
src/logging.py              StructuredLogger
src/memory/memory.py        MemoryStore, ConversationMemory, MemoryEntry
src/model/transformer.py    VoxlineTransformer, TransformerBlock, PositionalEncoding
src/planner/reasoning.py    Planner, ReasoningEngine, Plan, Step
src/providers/base.py       AIProvider ABC, GenerationConfig, ProviderHealth
src/providers/factory.py    ProviderFactory
src/providers/local_voxline.py      LocalVoxlineProvider
src/providers/local_transformers.py LocalTransformersProvider
src/tokenizer/bpe.py        BPETokenizer
src/tools/tools.py          ToolRegistry, Calculator, FileReadTool, FileWriteTool, DirectoryListTool
src/training/trainer.py     TrainingConfig, LanguageModelDataset, Trainer
```

### Scripts
```
scripts/train_small.py          Training script
scripts/expand_armenian.py      Armenian corpus expansion
scripts/expand_english_part[1-8].py  English corpus expansion
scripts/merge_english.py        English corpus merge
scripts/build_corpora.py        Corpus building
scripts/prepare_corpus.py       Corpus preparation
```

### Root Files
```
serve_v04.py        FastAPI server (v0.4)
chat.py             CLI chat interface
smoke_test.py       Phase 1 verification
baseline_smoke.py   Comprehensive component smoke tests
requirements.txt    Dependencies
```

### Checkpoints
```
checkpoints/v0_4/best_model.pt          Best model (epoch 9, 11.11MB)
checkpoints/v0_4/checkpoint_epoch_*.pt  Per-epoch checkpoints (10)
checkpoints/v0_4/tokenizer.json         BPE tokenizer
checkpoints/v0_4/config.json            Training config snapshot
checkpoints/v0_4/dataset_cache.pt       Cached tokenized dataset
```

### External Models
```
models/Qwen2.5-0.5B-Instruct/  Downloaded Qwen model (942MB, unused)
```

## Known Limitations

1. **Model quality**: 936K params is too small for coherent generation. Perplexity 135.8.
2. **Training data**: ~4,628 of 5,064 Armenian lines are template-generated, not natural text.
3. **No GPU**: CPU-only limits model size and training speed.
4. **No .gitignore**: Cache files, logs, and model files tracked in git.
5. **Legacy modules**: Top-level `src/model.py`, `src/tokenizer.py`, `src/config.py` shadowed by packages.
6. **Duplicate GenerationConfig**: Two different classes with different field names.
7. **Duplicate TrainingConfig**: Two different classes in different modules.
8. **Stub methods**: `Planner.decompose_task()` returns `[]`.
9. **No authentication**: API endpoints unauthenticated.
10. **No streaming**: FastAPI /chat doesn't support streaming responses.

## Git Status

- Branch: `main`
- 1 commit: `471abab feat: Voxline AI Core v0.3 stable`
- All v0.4 work is uncommitted
- No .gitignore exists
