# Phase 7 — Voxline Assistant MVP

**Status: PLANNING — DO NOT IMPLEMENT UNTIL APPROVED**

---

## 1. Objective

Transform Voxline AI Core from an AI infrastructure project into a usable AI assistant platform with three capabilities:

- **A. General AI Chat** — conversational assistant with memory
- **B. Business Intelligence** — domain-isolated business analysis
- **C. Coding Agent** — safe, bounded code-editing agent

The result should feel like an AI employee rather than a raw language model.

---

## 2. Current Architecture (As-Is)

```
USER
  |
  +-- CLI: chat.py (native model only)
  +-- API: serve_v04.py (provider-configurable)
  +-- API: src/api/server.py (business agent, legacy)
  |
PROVIDER LAYER (src/providers/)
  +-- AIProvider ABC
  +-- QwenProvider (current default)
  +-- LocalVoxlineProvider (research baseline)
  +-- ProviderFactory
  |
CONVERSATIONAL (src/api/chat.py)
  +-- ConversationalAI (text-only, uses TextGenerator, NOT AIProvider)
  |
MEMORY (src/memory/)
  +-- MemoryStore (SQLite, LIKE search)
  +-- ConversationMemory (rolling window)
  |
TOOLS (src/tools/)
  +-- ToolRegistry (4 tools: calculator, read_file, write_file, list_directory)
  +-- Workspace boundary: weak prefix check
  +-- No permission enforcement, no timeouts
  |
PLANNER (src/planner/)
  +-- Planner (stub: decompose_task returns [])
  +-- ReasoningEngine (stub: analyze_goal returns placeholder)
  |
AGENT (src/agent/)
  +-- AutonomousAgent (execution loop, hardcoded action dispatch)
  +-- States: IDLE/PLANNING/EXECUTING/OBSERVING/REVISING/VERIFYING/COMPLETED/FAILED
  +-- DOES NOT use AIProvider (requires model+tokenizer directly)
  |
BUSINESS (src/business/)
  +-- BusinessAgent (template-based plans, knowledge storage)
  +-- Plans live in RAM, no persistence
  |
CONFIGURATION (src/config/)
  +-- VoxlineConfig (flat string dict, env-driven)
  +-- No assistant-specific settings
  |
EVALUATION (src/evaluation/)
  +-- V2 with normalization and task-specific metrics
  +-- No Chat/Business/Coding benchmark categories
```

---

## 3. Target Architecture (To-Be)

```
USER
  |
  v
ASSISTANT API (single server)
  |
  v
VOXLINE ASSISTANT (src/assistant/)
  +-- mode routing (chat | business | coding)
  +-- session management
  +-- context building
  |
  +-- Chat Mode -----> ChatAssistant
  +-- Business Mode -> BusinessAssistant
  +-- Coding Mode ---> CodingAgent
  |
  v
SHARED INFRASTRUCTURE
  +-- AIProvider (via ProviderFactory)
  +-- MemoryStore + ConversationMemory
  +-- ToolRegistry (extended tools)
  +-- SessionManager
  |
  v
QWEN PROVIDER -> LOCAL QWEN MODEL
```

### Key Principles

1. **Single server** — one FastAPI app with assistant endpoints
2. **Single provider abstraction** — all modes use AIProvider, not direct model access
3. **Mode isolation** — chat/business/coding sessions do not share context
4. **Tool-driven coding** — coding agent operates through ToolRegistry, not raw shell
5. **Bounded autonomy** — iteration limits, workspace boundaries, execution policies
6. **Reuse first** — extend existing modules, do not create parallel systems

---

## 4. Modules to Reuse (No Changes Needed)

| Module | Path | Why Reusable |
|--------|------|--------------|
| AIProvider ABC | `src/providers/base.py` | Clean interface, works with Qwen |
| QwenProvider | `src/providers/qwen_provider.py` | Working, Phase 4 fixed |
| ProviderFactory | `src/providers/factory.py` | Configurable provider creation |
| MemoryStore | `src/memory/memory.py` | SQLite-backed, CRUD works |
| ConversationMemory | `src/memory/memory.py` | Rolling window, message storage |
| Evaluation framework | `src/evaluation/` | V2 metrics, runner, reports |
| VoxlineConfig | `src/config/settings.py` | Env-driven config |
| ModelConfig | `src/config/model_config.py` | Checkpoint compatibility |
| Error hierarchy | `src/errors.py` | Comprehensive exceptions |
| Checkpoint module | `src/checkpoint.py` | Save/load verified |

---

## 5. Modules to Extend

### 5.1. ToolRegistry (`src/tools/tools.py`)

**Current**: 4 tools, weak security, no execution tools.

**Additions needed**:
- `SearchFilesTool` — regex/glob search across workspace files
- `ApplyPatchTool` — apply unified diffs or search-and-replace edits
- `RunTestsTool` — execute pytest with structured output
- `RunCommandTool` — bounded shell execution (whitelist, timeout, cwd enforcement)

**Security hardening**:
- Replace `startswith()` boundary with `Path.is_relative_to()`
- Enforce permissions in `ToolRegistry.execute_tool()`
- Add execution timeout via threading
- Standardize return format: `{"ok": bool, "result": ..., "error": ...}`
- Add audit logging per tool call

### 5.2. BusinessAgent (`src/business/agent.py`)

**Current**: Template-based plans, no persistence, no AIProvider integration.

**Additions needed**:
- Accept `AIProvider` for LLM-driven analysis
- Persist plans to MemoryStore (not RAM dict)
- Add `analyze(request, context)` that uses provider for structured analysis
- Add `generate_report(request, context)` that produces formatted output
- Keep template fallback for when provider is unavailable

### 5.3. Planner/ReasoningEngine (`src/planner/reasoning.py`)

**Current**: Stubs (decompose_task returns [], analyze_goal returns placeholder).

**Additions needed**:
- `decompose_task(task)` — use AIProvider to generate step list
- `analyze_goal(goal)` — use AIProvider to assess complexity/tools
- Step actions must map to ToolRegistry names dynamically

### 5.4. MemoryStore (`src/memory/memory.py`)

**Current**: LIKE search, no FTS, no keyword auto-extraction.

**Minimum additions for Phase 7**:
- Auto-extract keywords in ConversationMemory.add_message()
- Basic FTS5 search (or at minimum, case-insensitive matching)
- No vector/embedding work (out of scope for Phase 7)

### 5.5. Error hierarchy (`src/errors.py`)

**Add**:
- `SessionError`, `SessionNotFoundError`
- `WorkspaceError`, `WorkspaceBoundaryError`
- `CodingAgentError`, `AgentPlanError`

### 5.6. Configuration (`src/config/settings.py`)

**Add defaults**:
```python
# Assistant
"ASSISTANT_NAME": "Voxline",
"ASSISTANT_DEFAULT_MODE": "chat",
"ASSISTANT_MAX_HISTORY": "20",

# Agent
"AGENT_MAX_ITERATIONS": "15",
"AGENT_STEP_TIMEOUT": "60",
"AGENT_EXECUTION_POLICY": "safe",  # safe|autonomous

# Workspace
"CODING_WORKSPACE_ROOT": ".",
"CODING_ALLOWED_COMMANDS": "python,pytest,pip,git",
"CODING_MAX_FILE_SIZE_MB": "10",
```

---

## 6. Modules to Create

### 6.1. `src/assistant/` — Assistant Core

```
src/assistant/
    __init__.py
    orchestrator.py    # Mode routing, session dispatch
    session.py         # Session data model, management
    context.py         # Context building (history + memory + system)
    chat.py            # Chat mode implementation
    business.py        # Business mode implementation
    coding.py          # Coding mode implementation
```

#### orchestrator.py

```python
class AssistantOrchestrator:
    """Routes requests to the correct capability mode."""
    
    def __init__(self, provider: AIProvider, memory_store: MemoryStore):
        self.provider = provider
        self.memory = memory_store
        self.sessions = SessionManager()
        self.chat = ChatAssistant(provider, memory_store)
        self.business = BusinessAssistant(provider, memory_store)
        self.coding = CodingAgent(provider, memory_store)
    
    async def handle(self, request: AssistantRequest) -> AssistantResponse:
        session = self.sessions.get_or_create(request.session_id, request.mode)
        if request.mode == "chat":
            return await self.chat.handle(session, request)
        elif request.mode == "business":
            return await self.business.handle(session, request)
        elif request.mode == "coding":
            return await self.coding.handle(session, request)
```

#### session.py

```python
@dataclass
class Session:
    session_id: str
    mode: str  # "chat" | "business" | "coding"
    created_at: str
    updated_at: str
    history: List[Dict[str, str]]
    metadata: Dict[str, Any]

class SessionManager:
    """In-memory session store with optional persistence."""
    
    def get_or_create(self, session_id: Optional[str], mode: str) -> Session: ...
    def get(self, session_id: str) -> Optional[Session]: ...
    def list_sessions(self) -> List[Session]: ...
    def delete(self, session_id: str) -> None: ...
```

#### context.py

```python
class ContextBuilder:
    """Builds provider-ready context from session + memory + system instructions."""
    
    def build(
        self,
        session: Session,
        user_message: str,
        system_prompt: str,
        memory_query: Optional[str] = None,
        max_history: int = 20,
    ) -> List[Dict[str, str]]:
        """
        Returns OpenAI-style messages list:
        [system, ...recent_history, user_message]
        
        Optionally retrieves relevant long-term memories
        and injects them into a system context message.
        """
```

#### chat.py

```python
class ChatAssistant:
    """General conversational assistant."""
    
    def __init__(self, provider: AIProvider, memory: MemoryStore): ...
    
    async def handle(self, session: Session, request: AssistantRequest) -> AssistantResponse:
        # 1. Build context from session history + memory
        # 2. Call provider.chat()
        # 3. Store user + assistant messages in session
        # 4. Optionally persist to long-term memory
        # 5. Return structured response
```

#### business.py

```python
class BusinessAssistant:
    """Business intelligence assistant with isolated context."""
    
    def __init__(self, provider: AIProvider, memory: MemoryStore): ...
    
    async def handle(self, session: Session, request: AssistantRequest) -> AssistantResponse:
        # 1. Retrieve business knowledge from memory
        # 2. Build business-specific context
        # 3. Call provider with business system prompt
        # 4. Store in business-tagged memory
        # 5. Return structured response
    
    async def analyze(self, topic: str, context: str) -> BusinessAnalysis: ...
    async def generate_report(self, topic: str, data: str) -> str: ...
```

#### coding.py

```python
class CodingAgent:
    """Safe coding agent with bounded execution."""
    
    def __init__(self, provider: AIProvider, memory: MemoryStore, 
                 workspace_root: str, execution_policy: str = "safe"): ...
    
    async def handle(self, session: Session, request: AssistantRequest) -> CodingResponse:
        # 1. Discover repository structure
        # 2. Plan task decomposition (using AIProvider)
        # 3. Execute steps through ToolRegistry
        # 4. Iterate within limits (edit -> test -> fix)
        # 5. Verify and report
    
    # State machine
    IDLE -> PLANNING -> INSPECTING -> EDITING -> TESTING -> FIXING -> VERIFYING -> COMPLETED/FAILED
```

---

## 7. API Design

**Single server**: evolve `serve_v04.py` into the canonical assistant API.

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | System health + provider status |
| `GET` | `/models` | Model info |
| `POST` | `/chat` | General chat (backward compatible) |
| `POST` | `/assistant/chat` | Chat mode with session management |
| `POST` | `/assistant/business` | Business intelligence mode |
| `POST` | `/assistant/code` | Coding agent mode |
| `GET` | `/assistant/sessions` | List active sessions |
| `GET` | `/assistant/sessions/{id}` | Get session details + history |
| `DELETE` | `/assistant/sessions/{id}` | Delete session |

### Request Models

```python
class AssistantRequest(BaseModel):
    message: str = Field(min_length=1, max_length=50000)
    session_id: Optional[str] = None  # auto-create if None
    max_tokens: int = Field(default=500, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)

class CodingRequest(AssistantRequest):
    workspace_root: Optional[str] = None  # override default
    execution_policy: str = Field(default="safe", pattern="^(safe|autonomous)$")
    max_iterations: int = Field(default=15, ge=1, le=50)
```

### Response Models

```python
class AssistantResponse(BaseModel):
    session_id: str
    mode: str
    response: str
    model: str
    metadata: Dict[str, Any] = {}

class CodingResponse(AssistantResponse):
    files_changed: List[str] = []
    tests_run: int = 0
    tests_passed: int = 0
    iterations: int = 0
    status: str  # "completed" | "failed" | "limit_reached"
```

---

## 8. Security Boundaries

### Workspace Containment

- All coding operations bounded to explicit `workspace_root`
- Use `Path.is_relative_to()` (Python 3.9+) for boundary checks
- No `..` traversal, no symlink escapes, no access outside root

### Command Execution

- Whitelist of allowed commands: `python`, `pytest`, `pip`, `git`
- `shell=False` always
- Forced `cwd=workspace_root`
- Enforced timeouts (default 60s per command)
- Stdout/stderr capture with byte caps (1MB)
- No environment variable passthrough (clean env)

### Execution Policies

**SAFE** (default):
- Plan and propose before executing
- Confirm before write_file, apply_patch, run_command
- Read operations are automatic

**AUTONOMOUS**:
- Execute permitted operations automatically
- Still obey workspace and command restrictions
- Audit log of all operations

### Input Limits

- Max message: 50,000 chars
- Max file read: 1MB
- Max file write: 1MB per operation
- Max search results: 100 files
- Max command output: 1MB

---

## 9. Coding Agent State Machine

```
IDLE
  |  (user submits task)
  v
PLANNING
  |  (AIProvider decomposes task into steps)
  v
INSPECTING
  |  (read relevant files, understand codebase)
  v
EDITING
  |  (apply changes via ToolRegistry)
  v
TESTING
  |  (run tests via RunTestsTool)
  v
  +-- tests pass --> VERIFYING --> COMPLETED
  |
  +-- tests fail --> FIXING --> EDITING (loop back)
  |
  +-- max iterations exceeded --> FAILED (report state)
```

### Iteration Limit

- Default: 15 iterations
- Configurable per request
- Each EDITING -> TESTING -> FIXING cycle = 1 iteration
- On limit: stop, preserve changes, report what was done and what remains

---

## 10. Testing Strategy

### Unit Tests

| Area | What to Test |
|------|-------------|
| `src/assistant/orchestrator.py` | Mode routing, session creation, error handling |
| `src/assistant/session.py` | CRUD, expiry, history management |
| `src/assistant/context.py` | Context building, memory injection, truncation |
| `src/assistant/chat.py` | Provider integration, history storage, memory persistence |
| `src/assistant/business.py` | Business context isolation, knowledge retrieval |
| `src/assistant/coding.py` | State machine transitions, iteration limits, boundary enforcement |
| `src/tools/tools.py` | New tools (search, patch, run_tests, run_command), security boundaries |
| `src/planner/reasoning.py` | LLM-driven decomposition, plan creation |

### Integration Tests

| Area | What to Test |
|------|-------------|
| Full chat flow | Request -> Orchestrator -> ChatAssistant -> Provider -> Response |
| Full business flow | Request -> BusinessAssistant -> Memory -> Provider -> Response |
| Full coding flow | Request -> CodingAgent -> Tools -> Provider -> Response |
| Session persistence | Create session -> Continue session -> Verify history |

### Security Tests

| Area | What to Test |
|------|-------------|
| Workspace boundary | Attempt path traversal, symlink escape, sibling dir access |
| Command whitelist | Attempt disallowed commands |
| Input limits | Oversized messages, files, command output |
| Session isolation | Business session doesn't leak into chat session |

### Regression

- All 205 existing tests must continue passing
- No tests deleted or weakened

### Test Infrastructure

- Create `tests/conftest.py` for shared fixtures (MockProvider, small model factory, temp workspace)
- Fix duplicate test class names (`TestNumberMatch`, `TestComputeCaseMetrics`)
- Gate heavy tests (Qwen) behind resource availability

---

## 11. Evaluation Benchmarks

Add new benchmark files (v1 preserved):

```
benchmarks/chat_en.jsonl      # English conversation benchmarks
benchmarks/chat_hy.jsonl      # Armenian conversation benchmarks
benchmarks/business_en.jsonl  # Business analysis benchmarks
benchmarks/coding_en.jsonl    # Coding task benchmarks
```

### Chat Benchmarks

- Multi-turn context retention
- Language detection and response
- System instruction following
- Memory-aware responses

### Business Benchmarks

- Business analysis with knowledge retrieval
- Report generation
- Strategic planning with context

### Coding Benchmarks

- File reading comprehension
- Bug identification
- Simple implementation tasks
- Test-driven development
- Tool-use correctness

---

## 12. Implementation Sequence

| Step | Module | Depends On | Estimated Complexity |
|------|--------|------------|---------------------|
| 1 | Configuration extensions | Nothing | Low |
| 2 | Error hierarchy extensions | Nothing | Low |
| 3 | Session management | Nothing | Medium |
| 4 | Context builder | Session, Memory | Medium |
| 5 | Chat assistant | Context, Provider | Medium |
| 6 | Business assistant | Chat, BusinessAgent | Medium |
| 7 | Tool hardening + new tools | Nothing | High |
| 8 | Coding agent state machine | Tools, Provider, Planner | High |
| 9 | Assistant orchestrator | Chat, Business, Coding | Medium |
| 10 | API integration | Orchestrator | Medium |
| 11 | Evaluation benchmarks | Evaluation framework | Medium |
| 12 | Security audit | Everything | High |
| 13 | Documentation | Everything | Medium |
| 14 | Final test suite + commit | Everything | Low |

---

## 13. Explicit Non-Goals

- **No fine-tuning** of Qwen or any model
- **No LoRA/QLoRA** implementation
- **No new model downloads**
- **No vector embeddings** or semantic search
- **No RAG** with external documents (knowledge stays in MemoryStore)
- **No authentication** or user management (single-user local)
- **No streaming** (provider doesn't support it well on CPU)
- **No GUI** — API + CLI only
- **No deletion of legacy code** — `src/legacy/` stays, old endpoints stay for backward compat
- **No rewriting Git history**

---

## 14. Acceptance Criteria

| # | Criterion | Verified By |
|---|-----------|-------------|
| 1 | Chat assistant works end-to-end | Integration test |
| 2 | Armenian chat works at provider level | Unit test + manual |
| 3 | English chat works | Integration test |
| 4 | Persistent conversation sessions work | Unit test |
| 5 | Memory integration works (context injected into prompts) | Unit test |
| 6 | Business AI mode works | Integration test |
| 7 | Business context is isolated from chat | Unit test |
| 8 | Coding agent can inspect repositories | Unit test |
| 9 | Coding agent can read files | Unit test |
| 10 | Coding agent can safely edit files | Unit test + security test |
| 11 | Coding agent can run tests | Unit test |
| 12 | Coding agent can diagnose failures | Unit test |
| 13 | Coding agent iterates within limit | Unit test |
| 14 | Workspace boundaries enforced | Security test |
| 15 | Dangerous operations controlled | Security test |
| 16 | Agent state transitions testable | Unit test |
| 17 | API integration works | Integration test |
| 18 | Evaluation benchmarks exist | Benchmark files present |
| 19 | Security tests pass | Security test suite |
| 20 | All 205 existing tests still pass | Full test suite |
| 21 | Documentation complete | docs/ files exist |
| 22 | No fine-tuning performed | Git history audit |
| 23 | No new model downloaded | Git history audit |
| 24 | Git history intact | Git log |

---

## 15. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Qwen 0.5B too weak for coding tasks | Coding agent produces poor plans | Keep iteration limit, structured prompts, graceful degradation |
| Memory search quality poor | Context injection ineffective | Start with keyword search, FTS5 if needed, no vector search |
| Tool security insufficient | Workspace escape | Conservative whitelist, `is_relative_to()`, extensive security tests |
| Session memory pressure | RAM exhaustion on long sessions | Cap session history, TTL expiry |
| State machine complexity | Bugs in coding agent loop | Explicit states, comprehensive unit tests, iteration hard limits |

---

## 16. Open Questions (For Discussion)

1. Should the coding agent support multi-file patches in a single step, or strictly one-file-at-a-time?
2. Should business mode auto-store all conversation turns, or only explicitly tagged knowledge?
3. Should we support session export/import for sharing coding sessions?
4. Should the assistant support a "research" mode for web search (future) or keep it strictly local?

---

*This plan was produced by a thorough audit of the entire Voxline AI Core repository. It identifies what exists, what needs extension, what needs creation, and what must not be changed. No code has been modified.*
