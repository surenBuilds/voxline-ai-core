# Voxline AI Core — Architecture

## System Overview

Voxline AI is a bilingual (Armenian + English) AI platform built around a custom decoder-only Transformer model, with a modular system layer for memory, tools, planning, agents, and API serving.

## Architecture Diagram

```
                         USER
                           |
                    +------+------+
                    |  API / CLI  |
                    +------+------+
                           |
                    +------+------+
                    | Orchestrator|
                    +------+------+
                           |
             +-------------+-------------+
             |             |             |
             v             v             v
         Memory         Planner        Tools
             |             |             |
             +-------------+-------------+
                           |
                    +------+------+
                    | AI Provider |
                    +------+------+
                           |
            +--------------+--------------+
            |                             |
            v                             v
      Native Voxline                    Qwen
         Model                         Backend
            |
            v
       Transformer
            |
         Tokenizer
```

## Package Structure (Canonical)

```
src/
  model/          VoxlineTransformer (decoder-only, 936K params)
  tokenizer/      BPE tokenizer (custom, 1109 vocab)
  training/       TrainingConfig, Trainer, Dataset, collate
  inference/      TextGenerator, GenerationConfig
  providers/      AIProvider ABC, LocalVoxlineProvider, LocalTransformersProvider, ProviderFactory
  attention/      ScaledDotProductAttention, MultiHeadAttention, CausalSelfAttention
  config/         VoxlineConfig (env-driven), ModelConfig (architecture), ModelType enum
  memory/         MemoryStore (SQLite), ConversationMemory
  tools/          ToolRegistry, Calculator, FileReadTool, FileWriteTool, DirectoryListTool
  planner/        Planner, ReasoningEngine, Plan, Step, PlanStatus
  agent/          AutonomousAgent, AgentState, ExecutionLog
  business/       BusinessAgent, BusinessPlan, BusinessPlanStep
  api/            FastAPI server, ConversationalAI
  evaluation/     (Phase 3 — placeholder)
  errors.py       Centralized error hierarchy (VoxlineError base)
  checkpoint.py   CheckpointLoader (save/load with config validation)
  logging.py      StructuredLogger, SecretFilteringFormatter, JSONFormatter
  legacy/         Archived v0.3 modules (reference only)
```

## Core Components

### Intelligence Layer

| Component | Location | Purpose |
|-----------|----------|---------|
| VoxlineTransformer | `src/model/transformer.py` | Decoder-only Transformer with causal attention, positional encoding |
| BPETokenizer | `src/tokenizer/bpe.py` | Byte-pair encoding with special tokens (`<PAD>`, `<UNK>`, `<BOS>`, `<EOS>`, `<CLS>`, `<SEP>`) |
| TextGenerator | `src/inference/generator.py` | Text generation with temperature, top-k, top-p, repetition penalty |
| ConversationalAI | `src/api/chat.py` | Multi-turn conversation with memory integration |

### System Layer

| Component | Location | Purpose |
|-----------|----------|---------|
| MemoryStore | `src/memory/memory.py` | SQLite-backed memory with search, typed memories, conversation history |
| ToolRegistry | `src/tools/tools.py` | Registered tools with schemas, permissions, execution |
| Planner | `src/planner/reasoning.py` | Task decomposition, plan creation, progress tracking |
| ReasoningEngine | `src/planner/reasoning.py` | Goal analysis, execution planning, revision decisions |
| AutonomousAgent | `src/agent/agent.py` | Orchestrates model + memory + tools + planner in execution loop |
| BusinessAgent | `src/business/agent.py` | Business-specific agent with knowledge storage |

### Provider Layer

| Component | Location | Purpose |
|-----------|----------|---------|
| AIProvider (ABC) | `src/providers/base.py` | Common interface: generate, stream, health_check |
| LocalVoxlineProvider | `src/providers/local_voxline.py` | Wraps native VoxlineTransformer |
| LocalTransformersProvider | `src/providers/local_transformers.py` | Wraps HuggingFace models (e.g. Qwen2.5-0.5B) |
| ProviderFactory | `src/providers/factory.py` | Creates providers from VoxlineConfig |

### Configuration

| Component | Location | Purpose |
|-----------|----------|---------|
| VoxlineConfig | `src/config/settings.py` | Environment-driven global config (provider, model path, device) |
| ModelConfig | `src/config/model_config.py` | Architecture config with checkpoint compatibility checking |
| TrainingConfig | `src/training/trainer.py` | Training hyperparameters |

### Infrastructure

| Component | Location | Purpose |
|-----------|----------|---------|
| CheckpointLoader | `src/checkpoint.py` | Model save/load with config validation |
| StructuredLogger | `src/logging.py` | Logging with secret filtering and JSON output |
| Error hierarchy | `src/errors.py` | Centralized exceptions (VoxlineError base class) |

## Error Hierarchy

```
VoxlineError
├── ModelError
│   ├── ModelLoadError
│   ├── ModelInferenceError
│   └── ModelConfigError
├── TokenizerError
│   ├── TokenizerLoadError
│   └── TokenizerEncodeError
├── CheckpointError
│   ├── CheckpointIncompatibilityError
│   └── CheckpointLoadError
├── ProviderError
│   ├── ProviderNotFoundError
│   └── ProviderUnavailableError
├── MemoryError
│   └── MemoryStoreError
├── ToolError
│   ├── ToolNotFoundError
│   ├── ToolExecutionError
│   └── ToolPermissionError
├── ConfigError
│   ├── ConfigLoadError
│   └── ConfigValidationError
├── TrainingError
│   └── TrainingDataError
└── AgentError
    ├── AgentTimeoutError
    └── AgentMaxIterationsError
```

## Data Flow

### Chat Request
```
User message
  -> FastAPI /chat endpoint
    -> ConversationalAI.chat()
      -> Context assembly (conversation history + memories)
      -> Prompt formatting ("User: {msg}\nAssistant:")
      -> TextGenerator.generate()
        -> Tokenization -> Model forward pass -> Token-by-token generation
      -> Response cleaning (strip prompt prefix, special tokens)
  -> JSON response
```

### Agent Execution
```
Goal
  -> AutonomousAgent.set_goal()
  -> Agent loop:
    -> ReasoningEngine.analyze_goal()
    -> Planner.create_plan()
    -> For each step:
      -> Execute via model or tool
      -> Observe result
      -> Verify progress
      -> Replan if needed
  -> Final result
```

## Model Specifications

| Parameter | v0.4 Small | Default |
|-----------|-----------|---------|
| Architecture | Decoder-only Transformer | Decoder-only Transformer |
| Parameters | 936,405 | ~175M (if scaled) |
| Hidden dim | 128 | 768 |
| Layers | 4 | 12 |
| Attention heads | 4 | 12 |
| FFN dim | 512 | 3072 |
| Max seq len | 128 | 2048 |
| Vocab size | 1,109 | 50,000 |
| Tokenizer | BPE | BPE |

## Legacy Modules

Legacy v0.3 modules are archived in `src/legacy/` for reference:

| Legacy File | Canonical Replacement |
|-------------|----------------------|
| `src/legacy/model.py` | `src/model/transformer.py` (VoxlineTransformer) |
| `src/legacy/tokenizer.py` | `src/tokenizer/bpe.py` (BPETokenizer) |
| `src/legacy/config.py` | `src/training/trainer.py` (TrainingConfig) |
| `src/legacy/dataset.py` | `src/training/trainer.py` (LanguageModelDataset) |
| `src/legacy/train.py` | `src/training/trainer.py` (Trainer) |
| `src/legacy/chat.py` | `src/api/chat.py` (ConversationalAI) |

Root-level legacy files (`main.py`, `train.py`, `generate.py`, `chat.py`) are v0.3 artifacts retained for reference.

## Known Issues

1. **Two GenerationConfig classes**: `src.inference.generator.GenerationConfig` (max_new_tokens) vs `src.providers.base.GenerationConfig` (max_tokens). Not merged because they serve different purposes (model-level vs provider-level).
2. **Planner.decompose_task()** is a stub returning `[]`.
3. **Model quality**: Current model (936K params) produces incoherent outputs. Perplexity ~135.8.
