# Phase 7 Step 11 Report — Tool Registration + End-to-End Coding Workflow

## Status: COMPLETE

## What was built

### Tool Bootstrap (`src/tools/bootstrap.py`)
- `build_tool_registry(config)` — single entry point that conditionally registers core, GitHub, Vercel, and workspace tools based on config and available credentials
- Core tools registered always: calculator, file_read, file_write, directory_list, execute_command
- GitHub tools registered only when `GITHUB_ENABLED=true` and token present
- Vercel tools registered only when `VERCEL_ENABLED=true` and token present
- Workspace tools registered when GitHub service is available

### Capability Discovery (`src/tools/tools.py`)
- `ToolRegistry.available_tools()` — returns categorized tool summaries safe for LLM context
- Excludes sensitive fields (credentials, tokens, security profiles)
- Returns tool name, description, category, and parameters for each tool

### CodingAgent Workflow Extensions (`src/assistant/coding.py`)
- `CodingTask` extended with: `repository_owner`, `repository_name`, `repository_branch`, `workspace_path`
- `execute_with_repository()` — end-to-end workflow with 8 phases:
  - **Phase A: Discovery** — validate repository context, check integration availability
  - **Phase B: Workspace** — clone repository, checkout branch
  - **Phase C: Review** — read codebase, identify files, understand structure
  - **Phase D: Plan** — LLM generates execution plan with available tools context
  - **Phase E: Execute** — three-phase tool execution (validate → authorize → execute)
  - **Phase F: Validate** — run tests, verify no regressions
  - **Phase G: GitHub** — create branch, commit changes, create PR
  - **Phase H: Vercel** — deploy preview (if enabled)
- `_exec_integration_tool()` — routes github_*/vercel_*/workspace_* actions to integration services
- `_build_plan_prompt()` — includes available tools context in LLM prompt

### API Endpoints (`serve_v04.py`)
- `POST /api/coding` — accepts CodingRequest (message, session_id, repository fields, PR/deploy options), returns CodingResult
- `GET /api/tools` — returns available tools by category (bootstrap-based)
- `GET /api/integrations` — returns integration status (never credentials)
- Fixed broken `/api/integrations` endpoint (wrong indentation, duplicate startup handler)

### UI (`src/api/static/index.html`)
- Coding mode button now active with full workflow support
- Repository owner, name, branch inputs
- "Create PR" and "Deploy Preview" checkboxes
- Response displays: files modified, PR URL, preview URL

### Security Regression Tests (Tests 26-30 in `test_tool_registration.py`)
- Path traversal blocked for coding workspace
- Unauthorized command blocked in coding context
- No tokens in tool schemas
- Audit log records coding operations
- Available tools has no sensitive data

### Smoke Test (`tests/smoke_external_integrations.py`)
- Gated by `VOXLINE_EXTERNAL_SMOKE` env var
- Tests real GitHub/Vercel auth access
- Never modifies production resources
- Requires real GITHUB_TOKEN and VERCEL_TOKEN

## Tests

### New Tests
- `tests/test_tool_registration.py` — 32 tests:
  - Bootstrap: 6 tests (core, github, vercel, workspace, no credentials, missing config)
  - Capability Discovery: 5 tests (has tools, categories, no tokens, no security profiles, core present)
  - Repository Context: 3 tests (fields, defaults, workflow types)
  - Full Workflow: 4 tests (execute_with_repository, all phases, metadata, workspace_path)
  - Integration Tool Execution: 4 tests (github, vercel, workspace, unknown)
  - API Endpoints: 4 tests (coding, tools, integrations, method not allowed)
  - Security Regression: 5 tests (path traversal, unauthorized command, no tokens in schemas, audit log, available_tools)
  - Smoke Test Gate: 1 test (skipped without VOXLINE_EXTERNAL_SMOKE)

- `tests/smoke_external_integrations.py` — 2 tests:
  - GitHub auth access (real)
  - Vercel auth access (real)

### Test Results
```
582 passed, 2 skipped (Windows symlink tests)
0 failures, 0 regressions
```

## Security

| Requirement | Status |
|-------------|--------|
| No auto PR merge | ✅ |
| No repo deletion | ✅ |
| No production deploy without approval | ✅ |
| No unrestricted shell/network | ✅ |
| No credential extraction | ✅ |
| No self-modifying security | ✅ |
| No token leakage to LLM/logs/API | ✅ |
| Tool schemas safe for LLM context | ✅ |
| Workspace boundary enforced | ✅ |
| Command policy enforced | ✅ |
| Audit trail for all operations | ✅ |

## Files Changed

| File | Status |
|------|--------|
| `src/tools/bootstrap.py` | NEW |
| `src/tools/tools.py` | MODIFIED (added `available_tools()`) |
| `src/tools/__init__.py` | MODIFIED (export `build_tool_registry`) |
| `src/assistant/coding.py` | MODIFIED (extended CodingTask, phases A-H, integration tools) |
| `serve_v04.py` | REWRITTEN (fixed broken code, added /api/coding, /api/tools) |
| `src/api/static/index.html` | MODIFIED (coding mode UI) |
| `tests/test_tool_registration.py` | NEW |
| `tests/smoke_external_integrations.py` | NEW |

## Known Limitations

- RepositoryWorkspace uses raw subprocess for git operations — no credential helper for private repos
- Vercel deployment monitoring (polling for status changes) is not implemented
- OAuth/App authentication architecture is prepared but not implemented (env token only)
- No automatic PR merge capability (by design)
- LLM planning quality depends on Qwen2.5-0.5B capability (limited)

## Git

- commit: NOT CREATED
- ready for review: YES
