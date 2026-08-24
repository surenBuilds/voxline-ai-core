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
              +------------+------------+
              |            |            |
              v            v            v
         ChatAssistant  Business     CodingAgent
              |         Assistant        |
              +------+  |  +------+     |
                     |  |  |      |     |
                     v  v  v      v     v
                 ContextBuilder  Workspace
                     |
              +------+------+
              |   Memory    |
              +------+------+
                     |
              +------+------+
              | AI Provider |
              +------+------+
                     |
          +----------+----------+
          |                     |
          v                     v
    Native Voxline            Qwen
       Model                Backend
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
  providers/      AIProvider ABC, ModelInfo, LocalVoxlineProvider, QwenProvider, ProviderFactory
  attention/      ScaledDotProductAttention, MultiHeadAttention, CausalSelfAttention
  config/         VoxlineConfig (env-driven), ModelConfig (architecture), ModelType enum
  memory/         MemoryStore (SQLite), ConversationMemory
  tools/          ToolRegistry, Calculator, FileReadTool, FileWriteTool, DirectoryListTool
  planner/        Planner, ReasoningEngine, Plan, Step, PlanStatus
  agent/          AutonomousAgent, AgentState, ExecutionLog
  business/       BusinessAgent, BusinessPlan, BusinessPlanStep
  api/            FastAPI server, ConversationalAI
  assistant/      ChatAssistant, BusinessAssistant, ContextBuilder, Session, SessionManager (Phase 7)
  evaluation/     Schemas, metrics, datasets, runner, reports, comparison, normalize
  errors.py       Centralized error hierarchy (VoxlineError base)
  checkpoint.py   CheckpointLoader (save/load with config validation)
  logging.py      StructuredLogger, SecretFilteringFormatter, JSONFormatter
  legacy/         Archived v0.3 modules (reference only)
```

## Core Components

### Assistant Layer (Phase 7)

| Component | Location | Purpose |
|-----------|----------|---------|
| ChatAssistant | `src/assistant/chat.py` | Main conversational interface — orchestrates session, context, provider |
| BusinessAssistant | `src/assistant/business.py` | Business intelligence — analysis, strategy, KPIs, action planning |
| BusinessContext | `src/assistant/business.py` | Typed business context: company, industry, KPIs, goals |
| BusinessRequest | `src/assistant/business.py` | Structured request with task_type, message, context, language |
| BusinessResponse | `src/assistant/business.py` | Structured response with text, recommendations, action items, risks |
| BusinessPlan | `src/assistant/business.py` | Structured plan: objective, strategy, action items, risks, metrics |
| KPI | `src/assistant/business.py` | Key Performance Indicator with value, target, unit, period |
| Recommendation | `src/assistant/business.py` | Strategic recommendation with rationale, impact, effort, risk |
| ActionItem | `src/assistant/business.py` | Actionable step with priority, dependencies, expected outcome |
| ContextBuilder | `src/assistant/context.py` | Assembles structured context (memory, history, mode) for provider input |
| Session | `src/assistant/session.py` | Isolates conversation state by mode (chat/business/coding) |
| SessionManager | `src/assistant/session.py` | In-memory session store with eviction and mode isolation |
| AssistantResponse | `src/assistant/chat.py` | Structured response with text, session, provider, and metadata |

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
| AIProvider (ABC) | `src/providers/base.py` | Common interface: chat, generate, stream, health_check, get_model_info |
| **ModelInfo** | `src/providers/base.py` | Dataclass: model_id, provider_id, parameters, vocab_size, supports_streaming |
| LocalVoxlineProvider | `src/providers/local_voxline.py` | Wraps native VoxlineTransformer, streaming support |
| **QwenProvider** | `src/providers/qwen_provider.py` | Wraps local Qwen2.5 via HuggingFace transformers, chat template |
| ProviderFactory | `src/providers/factory.py` | Creates providers from VoxlineConfig, lazy registration |

### Evaluation Layer

| Component | Location | Purpose |
|-----------|----------|---------|
| BenchmarkCase | `src/evaluation/schemas.py` | Benchmark case with prompt, expected answer, category, language |
| EvalReport | `src/evaluation/schemas.py` | Complete evaluation report with per-case results and summaries |
| EvaluationStatus | `src/evaluation/schemas.py` | PASS, PARTIAL, FAIL, INVALID_EVALUATION status enum |
| HumanEvalScores | `src/evaluation/schemas.py` | Structured human evaluation with notes field (Phase 6) |
| Metrics | `src/evaluation/metrics.py` | 20+ metric functions: task-specific, smart_contains, normalized_match |
| Normalize | `src/evaluation/normalize.py` | Text normalization: Unicode, whitespace, numbers, Armenian-specific |
| Datasets | `src/evaluation/datasets.py` | JSONL benchmark loading, saving, filtering |
| EvaluationRunner | `src/evaluation/runner.py` | Orchestrates provider evaluation with task-specific pass logic |
| Reports | `src/evaluation/reports.py` | Text formatting, JSON save/load for reports |
| Comparison | `src/evaluation/comparison.py` | Run comparison, regression detection |

### Configuration

| Component | Location | Purpose |
|-----------|----------|---------|
| VoxlineConfig | `src/config/settings.py` | Environment-driven global config (provider=qwen default, model path, device) |
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
│   ├── ToolPermissionError
│   └── CommandDeniedError
├── ConfigError
│   ├── ConfigLoadError
│   └── ConfigValidationError
├── TrainingError
│   └── TrainingDataError
├── AgentError
│   ├── AgentTimeoutError
│   └── AgentMaxIterationsError
├── SessionError
│   ├── SessionNotFoundError
│   └── SessionExpiredError
├── WorkspaceError
│   └── WorkspaceBoundaryError
└── CodingAgentError
    └── AgentPlanError
```

## Data Flow

### Assistant Chat (Phase 7)
```
User message
  -> ChatAssistant.chat(session_id, message)
    -> SessionManager.get(session_id)
    -> ContextBuilder.build(session, message)
      -> MemoryStore.search_memories(query)  ← memory actually reaches the model
      -> Format session history (capped by budget)
      -> Add mode instruction
      -> Assemble ordered messages
    -> AIProvider.chat(messages, config)
      -> Provider handles system instruction + generation
    -> Session.add_message(user)
    -> Session.add_message(assistant)
    -> Optional: MemoryStore.add_memory() (only if explicitly useful)
  -> AssistantResponse
```

### Business Intelligence (Phase 7)
```
BusinessRequest
  -> BusinessAssistant.analyze(session_id, request)
    -> SessionManager.get(session_id)
    -> Validate session mode = BUSINESS
    -> Build business context string from BusinessContext
    -> ContextBuilder.build(session, message, mode_instructions, business_context)
      -> MemoryStore.search_memories(query)
      -> Format session history
      -> Add business-specific mode instruction
      -> Assemble ordered messages
    -> AIProvider.chat(messages, config)
    -> Session.add_message(user)
    -> Session.add_message(assistant)
    -> Optional: MemoryStore.add_memory() (business-tagged)
  -> BusinessResponse
```
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
4. **QwenProvider**: Does not support streaming (HuggingFace generate API is synchronous). Streaming support requires custom generation loop.
5. **Qwen Armenian capability**: Qwen2.5-0.5B scores 0% on Armenian benchmarks. Limited Armenian vocabulary and instruction following.
6. **Transformer API compatibility**: transformers 5.15.0 changed `apply_chat_template()` to return `BatchEncoding` instead of a tensor. QwenProvider now handles both formats. `torch_dtype` renamed to `dtype`.

## Provider Interface (Phase 2)

All providers implement the `AIProvider` ABC defined in `src/providers/base.py`:

```python
class AIProvider(ABC):
    @property
    def provider_id(self) -> str: ...       # e.g. "local_voxline", "qwen"
    @property
    def model_id(self) -> str: ...          # e.g. "voxline_0.4.0", "Qwen2.5-0.5B-Instruct"
    @property
    def supports_streaming(self) -> bool: ...

    def get_model_info(self) -> ModelInfo: ...   # returns ModelInfo dataclass
    async def health_check(self) -> ProviderHealth: ...
    async def generate(self, prompt, config) -> str: ...   # default: calls stream()
    async def chat(self, messages, config) -> str: ...     # default: calls generate()
    async def stream(self, prompt, config) -> AsyncIterator[str]: ...  # default: raises NotImplementedError
```

Provider selection is driven by `VoxlineConfig.ai_provider` ("native" or "qwen").
The server accepts `--provider` flag: `python serve_v04.py --provider qwen`.
