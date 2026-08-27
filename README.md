# Voxline AI Core

Local-first AI core with a hands-off **coding agent** that can plan, edit, test,
commit, and open PRs in a repository — plus chat, business intelligence, a BPE
tokenizer, a transformer model, an evaluation framework, and GitHub/Vercel
integrations.

**Current version: v1.0.0.** See [docs/V1_RELEASE_REPORT.md](docs/V1_RELEASE_REPORT.md).

## Quick Start

```powershell
pip install -r requirements.txt
# Optional: populate src/config settings via environment (see .env.example)

# Serve the API (default provider=qwen — requires a local Qwen2.5 model)
python serve_v04.py --provider qwen --port 8000

# Interactive chat
python chat.py
```

## What This Repo Does

- **Coding agent** (`src/assistant/coding.py`) — the flagship feature. An
  autonomous 8-phase (A-H) workflow: accelerate code → plan → execute → validate →
  fix-loop → commit → push → open a PR → optionally deploy. It uses the secured
  `ToolRegistry` (path/command security + audit log) for every file write and
  command.
- **Integration tools** (`src/tools/integration_tools.py`) — GitHub
  (read/branch/commit/PR) and Vercel (deploy), gated by permission policies.
- **Chat / Business** assistants, sessions, context builder, memory.
- **Model stack**: BPE tokenizer, VoxlineTransformer (936K params), Qwen2.5
  provider wrapper, training (`train.py`), evaluation (`evaluate.py`).

## Run the Tests

```powershell
# Default unit/integration suite (no network, no credentials)
python -m pytest tests/ -q --ignore=tests/test_real_e2e_validation.py

# Real end-to-end validation against a local git repo (no network)
$env:VOXLINE_REAL_E2E="1"; python -m pytest tests/test_real_e2e_validation.py -q

# Real external smoke (only with VOXLINE_EXTERNAL_SMOKE=1 + real credentials)
$env:VOXLINE_EXTERNAL_SMOKE="1"; python -m pytest tests/smoke_real_integrations.py -q
```

The real E2E suite needs `git` and `pytest` on PATH. It builds its own temporary
bare remote and clone, uses a deterministic scripted provider, and then cleans up.

## Documentation

| Doc | What it covers |
|-----|----------------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Canonical module map, architecture |
| [docs/CODING_AGENT_AUDIT.md](docs/CODING_AGENT_AUDIT.md) | Coding agent design & audit |
| [docs/PHASE7_PLAN.md](docs/PHASE7_PLAN.md) | Phase 7 roadmap (includes agent + integrations) |
| [docs/GITHUB_INTEGRATION.md](docs/GITHUB_INTEGRATION.md), [docs/VERCEL_INTEGRATION.md](docs/VERCEL_INTEGRATION.md) | Integrations |
| [docs/DEVELOPMENT_STATUS.md](docs/DEVELOPMENT_STATUS.md) | Live status, component table, known limitations |
| [docs/V1_RELEASE_REPORT.md](docs/V1_RELEASE_REPORT.md) | v1.0.0 release report (real E2E + fixes) |

## Security Model

Every tool invocation passes a three-phase check (validate → authorize → audit).
The `ToolRegistry` enforces a workspace boundary (`PathSecurity`), an allowed
command policy, and writes an `AuditLog`. GitHub/Vercel tokens are never exposed
to the LLM; destructive operations always require approval; PRs are never
auto-merged and production deploys are blocked without approval.

## Notable Conventions

- All intelligence flows through an `AIProvider` ABC — never direct model calls.
- Tests are real and reproducible with deterministic scripted providers; suites
  that need live models/credentials are gated behind env vars and skip otherwise.
- Runtime state (`memory/*.db`, `__pycache__`, logs, `*.pth`) is gitignored and
  never committed.
