# CodingAgent Repository Audit

Date: 2026-08-25
Commit: b388fe8

## Existing Components — Reusable

| Component | Location | Reuse |
|-----------|----------|-------|
| PathSecurity | `src/tools/security.py:142` | YES — workspace boundary enforcement |
| FileSizeGuard | `src/tools/security.py:207` | YES — file size limits for read/write |
| CommandPolicy | `src/tools/security.py:254` | YES — allowed/denied/approval commands |
| CommandValidator | `src/tools/security.py:347` | YES — safe command parsing + execution |
| AuditLog | `src/tools/security.py:85` | YES — append-only tool audit trail |
| ToolSecurityProfile | `src/tools/security.py:52` | YES — capability declarations |
| PermissionDecision | `src/tools/security.py:31` | YES — ALLOWED/DENIED/REQUIRES_APPROVAL |
| ToolRegistry | `src/tools/tools.py:293` | YES — three-phase API (validate/authorize/execute) |
| FileReadTool | `src/tools/tools.py:98` | YES — safe file read within workspace |
| FileWriteTool | `src/tools/tools.py:159` | YES — safe file write within workspace |
| DirectoryListTool | `src/tools/tools.py:210` | YES — safe directory listing |
| CommandExecutor | `src/tools/tools.py:252` | YES — safe command execution |
| AIProvider | `src/providers/base.py:60` | YES — ABC for all providers |
| GenerationConfig | `src/providers/base.py:25` | YES — generation parameters |
| ContextBuilder | `src/assistant/context.py:88` | YES — with language_instruction support |
| Session/SessionManager | `src/assistant/session.py` | YES — SessionMode.CODING exists |
| VoxlineConfig | `src/config/settings.py` | YES — CODING_* settings exist |
| MemoryStore | `src/memory/memory.py` | YES — optional memory integration |
| CodingAgentError | `src/errors.py:151` | YES — base error |
| AgentPlanError | `src/errors.py:155` | YES — plan parse failure |
| WorkspaceError | `src/errors.py:142` | YES — workspace errors |
| WorkspaceBoundaryError | `src/errors.py:146` | YES — boundary violations |
| CommandDeniedError | `src/errors.py:159` | YES — denied commands |
| AgentTimeoutError | `src/errors.py:120` | YES — execution timeout |
| AgentMaxIterationsError | `src/errors.py:124` | YES — iteration limit |

## Conflicts — Do Not Reuse

| Component | Location | Reason |
|-----------|----------|--------|
| AutonomousAgent | `src/agent/agent.py` | Takes raw model/tokenizer, not AIProvider. Legacy. |
| Planner | `src/planner/reasoning.py` | Basic stub, no LLM-driven planning. |

## Architecture Decision

- New CodingAgent goes in `src/assistant/coding.py`
- Does NOT modify AutonomousAgent or Planner
- Uses AIProvider for all LLM interaction
- Uses ToolRegistry three-phase API for all tool calls
- Uses PathSecurity for all workspace boundary enforcement
- Uses CommandPolicy/CommandValidator for all command execution
- Uses AuditLog for complete audit trail

## Gap Analysis

| Need | Status |
|------|--------|
| File search (grep-like) | MISSING — need FileSearchTool |
| Project structure inspection | MISSING — need ProjectInspector tool or method |
| LLM structured plan generation | MISSING — need CodingAgent to call AIProvider |
| Bounded fix loop | MISSING — need iteration tracking |
| Approval workflow | PARTIAL — PermissionDecision exists, need CodingAgent integration |

## Existing Config for Coding

```
CODING_WORKSPACE_ROOT = "."
CODING_ALLOWED_COMMANDS = "python,pytest,pip,git"
CODING_MAX_FILE_SIZE_MB = 10
CODING_MAX_OUTPUT_BYTES = 1048576
AGENT_MAX_ITERATIONS = 15
AGENT_STEP_TIMEOUT = 60
```

Need to add:
- CODING_AGENT_MAX_PLAN_STEPS (default: 10)
- CODING_AGENT_MAX_CONTEXT_CHARS (default: 8000)
- CODING_AGENT_MAX_FIX_ITERATIONS (default: 3)
- CODING_AGENT_REQUIRE_APPROVAL_FOR_WRITES (default: true)
