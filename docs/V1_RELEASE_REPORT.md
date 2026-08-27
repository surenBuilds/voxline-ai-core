# Voxline AI Core — v1.0.0 Release Report

## Date: 2026-08-27
## Version: v1.0.0
## Scope: Phase 7 Steps 13-14 (finalize coding agent + production readiness)

## Summary

v1.0.0 finalizes the Voxline AI Core Coding Agent. Step 12 delivered production
hardening; Step 13 added **real end-to-end validation** against a real local git
repository using a deterministic scripted provider (no hosted credentials or
network required), which surfaced and fixed four real workflow defects. Step 14
verified production readiness and finalized the release.

No redesign or speculative features were added. Only real blockers were fixed,
each covered by a regression test.

## Step 13 — Real End-to-End Validation

Five real scenarios drive the REAL `CodingAgent` / `ToolRegistry` / git pipeline
against a real local git repository (`tests/demo_repo.py` seeds a bare remote +
working clone with an intentional bug and failing tests). A deterministic
`ScriptedProvider` (`tests/test_real_e2e_validation.py`) supplies valid JSON
plans so validation is reproducible while still exercising the full real agent
plumbing: discovery → workspace → planning → execution → test → commit → push → PR.

| # | Scenario | Result |
|---|----------|--------|
| 1 | Simple bug fix (one file, verify only target changed) | PASS |
| 2 | Small feature (multi-file, existing + new tests pass) | PASS |
| 3 | Intentional failure (authorization denial classified safely, no leaks) | PASS |
| 4 | GitHub workflow (local git: commit/push separate from PR, branch sanitized, clean working tree) | PASS |
| 5 | UI → API → Agent (real FastAPI + TestClient, safe errors, operation_id) | PASS |

Gated behind `VOXLINE_REAL_E2E=1` because they shell out to git and pytest. Run:

```powershell
$env:VOXLINE_REAL_E2E="1"
python -m pytest tests/test_real_e2e_validation.py -v
```

## Real Defects Found & Fixed (each with a regression test)

| # | Severity | Root cause | Fix |
|---|----------|-----------|-----|
| 1 | CRITICAL | `_phase_workspace` replaced `self.tool_registry` with a fresh empty registry, silently dropping all GitHub/Vercel/workspace tools, so PR creation and deployment phases lost their tools. | `ToolRegistry.set_workspace_root()` re-roots the security boundary while PRESERVING all registered tools; `_phase_workspace` calls it (`src/tools/tools.py`, `src/assistant/coding.py`). |
| 2 | HIGH | `_phase_commit_and_push` constructed `RepositoryWorkspace(self.workspace, …)` but `self.workspace` was already the nested repo dir `root/{owner}__{repo}` → double nesting → `NotADirectoryError`. | Added `CodingAgent.workspace_root` (immutable original root) and use it for `RepositoryWorkspace` construction (`src/assistant/coding.py`). |
| 3 | HIGH | Agent never created the feature branch, so all work landed on the base branch and the final feature-branch push failed ("refspec does not match any"). | `_phase_workspace` now creates + checks out the feature branch after clone (`src/assistant/coding.py`). |
| 4 | MEDIUM | `RepositoryWorkspace._run_git()`/`clone()` called `AuditLog.log_event()` which did not exist → `AttributeError` whenever a log was supplied. | Added `AuditLog.log_event()` recording generic framework events (`src/tools/security.py`). |

## Step 14 — Production Readiness

- Full automated test gate passes (see below).
- `ToolRegistry.set_workspace_root()` change covered by regression tests; existing
  suites (coding agent, tool registration, security, E2E workflow) re-verified — no regressions.
- `.gitignore` extended to cover runtime state (`*.db`, `memory/`, `*.sqlite*`),
  runtime logs/outputs, and model stubs (`*.pth`); tracked runtime artifacts and
  scratch files removed from the release tree.
- Scratch development files removed / not included: `tests/armenian_direct_test.py`,
  `tests/diagnose_server.py`, `tests/smoke_server.py`. Real reproducible
  automated tests kept: `tests/test_real_e2e_validation.py`, `tests/demo_repo.py`.

## Files Changed / Created (v1.0 final)

| File | Change |
|------|--------|
| `src/assistant/coding.py` | `workspace_root` attribute; `_phase_workspace` uses `set_workspace_root()` + creates feature branch; commit/push uses `workspace_root` |
| `src/tools/tools.py` | New `ToolRegistry.set_workspace_root()` |
| `src/tools/security.py` | New `AuditLog.log_event()`; new `CommandValidator.set_path_security()` |
| `tests/demo_repo.py` | NEW — local bare remote + seeded demo clone |
| `tests/test_real_e2e_validation.py` | NEW — 5 real E2E scenarios (gated) |
| `tests/test_production_hardening.py` | +3 regression tests (tests 40-42) for the fixed defects |
| `.gitignore` | Runtime state / logs / model stubs ignored |
| `docs/DEVELOPMENT_STATUS.md` | Updated to v1.0.0 status, test table, fixes, git history |
| `docs/V1_RELEASE_REPORT.md` | This report |
| `README.md` | NEW — developer orientation |

## Test Results (final gate)

### Default suite (no gated env vars)

| Suite | Result |
|-------|--------|
| test_architecture, test_coding_agent, test_evaluation, test_language, test_tools_security, test_production_hardening | PASS |
| test_core, test_assistant, test_assistant_business, test_assistant_chat, test_language, test_providers, test_server | PASS |
| test_assistant_context, test_business_agent, test_armenian_benchmark | PASS |
| test_github_integration, test_vercel_integration | PASS |
| test_tool_registration, e2e_coding_workflow | PASS |

### Gated real E2E (`VOXLINE_REAL_E2E=1`)

| Suite | Result |
|-------|--------|
| test_real_e2e_validation.py (5 scenarios) | **PASS** |

## Security Summary (unchanged from Step 12, still holds)

- Path traversal: PROTECTED
- Command injection: PROTECTED
- Token leakage: PROTECTED
- Branch name injection: PROTECTED
- PR auto-merge: NEVER attempted
- Production deploy: BLOCKED without approval
- API error leakage: PROTECTED

## Architectural Invariants Maintained

- All intelligence flows through Assistant → AIProvider → QwenProvider
- No direct Qwen instantiation
- No fine-tuning or new model downloads
- No Git history rewrite
- All tool calls audited with operation_id traceability
- Integration tools are NEVER dropped when the workspace boundary is re-rooted
