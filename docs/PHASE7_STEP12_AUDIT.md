# Phase 7 Step 12 — Audit Report

## Current Architecture

```
User → Web UI → FastAPI (/api/coding) → CodingAgent → ToolRegistry → Tools → OS/GitHub/Vercel
                                          ↓
                                    AIProvider → Qwen2.5
                                          ↓
                                    ContextBuilder → MemoryStore
```

## Verified Functionality

| Component | Status | Notes |
|-----------|--------|-------|
| Core tool registration | ✅ | calculator, read_file, write_file, list_directory, execute_command |
| GitHub tool registration | ✅ | 6 tools, conditional on GITHUB_ENABLED + token |
| Vercel tool registration | ✅ | 3 tools, conditional on VERCEL_ENABLED + token |
| Workspace tool registration | ✅ | clone, diff, test |
| Capability discovery | ✅ | available_tools() returns categorized summaries |
| Three-phase API | ✅ | validate → authorize → execute with audit |
| CodingAgent local execution | ✅ | Plan → execute → validate → fix loop |
| CodingAgent repository workflow | ✅ | Phases A-H |
| Web UI Chat/Business/Coding | ✅ | All modes functional |
| API endpoints | ✅ | /api/chat, /api/business, /api/coding, /api/tools, /api/integrations |
| Credential management | ✅ | EnvironmentCredentialProvider, redact(), never exposed |
| Permission policies | ✅ | GitHub READ/WRITE/DESTRUCTIVE, Vercel PREVIEW/PRODUCTION |
| Audit logging | ✅ | Every tool invocation recorded |

## Known Defects

### CRITICAL

1. **`_phase_github` never commits/pushes code** (`coding.py:482-537`)
   - Calls `workspace_diff` to read diff, then calls `github_create_pull_request`
   - Never executes `git add`, `git commit`, `git push`
   - PR creation will always fail because no changes are committed
   - FIX: Add workspace_commit + workspace_push calls before PR creation

2. **RepositoryWorkspace bypasses ToolRegistry security** (`integration_tools.py:37-170`)
   - `clone()`, `checkout()`, `create_branch()`, `diff()`, `status()`, `commit()`, `push()`, `run_tests()` all use raw `subprocess.run()`
   - No CommandValidator, no AuditLog, no PathSecurity for working directory
   - FIX: Route through CommandValidator or add security checks

### HIGH

3. **`require_approval_for_writes` blocks ALL file modifications** (`coding.py:921-928`)
   - When enabled (default: true), every write action is SKIPPED
   - The agent can never actually modify files
   - FIX: Add auto-approve for coding agent workspace writes (within workspace boundary)

4. **Vercel deployment uses wrong branch** (`coding.py:539-560`)
   - Passes `task.repository_branch` (base branch) to deployment, not the feature branch
   - FIX: Pass feature_branch to Vercel deployment

5. **No deployment verification** — Vercel deployment is created but never polled for READY/ERROR/CANCELED status
   - FIX: Add polling with configurable interval and timeout

### MEDIUM

6. **`available_tools()` missing `requires_approval`** — Spec requires this field
   - FIX: Add requires_approval field from ToolSecurityProfile

7. **No operation_id in CodingResult** — No traceability between request and audit entries
   - FIX: Generate operation_id, pass through to audit entries

8. **CodingResult missing fields** — Needs `status`, `tests_passed`, `tests_failed`, `commit`
   - FIX: Extend CodingResult dataclass

9. **API error responses leak internal details** — `f"{type(exc).__name__}: {exc}"` exposes exception types
   - FIX: Return generic error messages, log details server-side

10. **No request timeout on coding endpoint** — Long-running tasks block FastAPI
    - FIX: Run in thread with timeout

11. **Branch name not sanitized** — `voxline/{task_id}` could contain invalid chars
    - FIX: Sanitize to alphanumeric + hyphens only

12. **RepositoryWorkspace.run_tests uses `test_command.split()`** — potential injection vector
    - FIX: Use shlex.split or validate command

### LOW

13. **Traceback printed in API error handlers** — `traceback.print_exc()` on every error
    - FIX: Use logger.exception() instead

14. **`_categorize_tools` in bootstrap.py duplicates logic** from `available_tools()` in tools.py
    - Minor duplication, acceptable

## Risks

- Large codebase changes needed for security fixes (RepositoryWorkspace)
- Deployment verification adds complexity and timeout handling
- Thread safety of CodingAgent (called from async FastAPI endpoint)

## Proposed Changes

| Step | Files Affected | Priority |
|------|---------------|----------|
| 12.1 Config validation | settings.py, credentials.py, tests | HIGH |
| 12.2 Tool capabilities | tools.py, tests | HIGH |
| 12.3 GitHub workflow | coding.py, integration_tools.py, tests | CRITICAL |
| 12.4 Failure recovery | coding.py, tests | HIGH |
| 12.5 Test safety | integration_tools.py, tests | HIGH |
| 12.6 Vercel verification | coding.py, integration_tools.py, tests | HIGH |
| 12.7 E2E dry run | tests/e2e_coding_workflow.py | HIGH |
| 12.8 Smoke test | tests/smoke_real_integrations.py | MEDIUM |
| 12.9 Web UI | index.html | HIGH |
| 12.10 API audit | serve_v04.py | HIGH |
| 12.11 Observability | coding.py, security.py | HIGH |
| 12.12 Security audit | docs/ | HIGH |
| 12.13 Test suite | all tests | HIGH |
| 12.14 Docs + commit | docs/, git | HIGH |
