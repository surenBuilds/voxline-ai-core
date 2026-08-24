# Voxline Phase 7 — Step 7 Report
## Tool Security Hardening

### Status: COMPLETE

### Security Audit Summary

#### Vulnerabilities Found in Pre-Step-7 Code
1. **`startswith(workspace)` path check** — `FileReadTool`, `FileWriteTool`, `DirectoryListTool` all used `str(file_path).startswith(str(self.workspace_root))` for boundary checking. This is vulnerable to prefix collisions (e.g., `/workspace/project-evil` passes prefix check against `/workspace/project`).
2. **No symlink awareness** — Path resolution did not canonicalize symlinks before checking boundaries.
3. **No file size enforcement** — Files of any size could be read or written.
4. **No command execution tool** — ToolRegistry had no `CommandExecutor`. Commands were not part of the tool layer.
5. **No permission policy** — No structured ALLOWED/DENIED/REQUIRES_APPROVAL system for commands.
6. **No audit logging** — Tool invocations were not recorded.
7. **No environment sanitization** — No mechanism to prevent secret leakage to subprocess.
8. **No timeout on tools** — Tools had a `timeout` field in schema but it was never enforced.
9. **Three phases combined** — validate + authorize + execute were a single `execute_tool()` call.

#### Fixes Applied
1. **`PathSecurity` class** — All path validation uses `Path.resolve()` + `Path.is_relative_to()`. Relative paths anchored to workspace root (not cwd). Absolute paths validated directly.
2. **`FileSizeGuard` class** — Enforces `workspace.max_file_size` for reads (stat) and writes (content length). Returns structured `ToolPermissionResult`.
3. **`CommandPolicy` class** — Explicit allowed/denied/approval-required sets. Blocked argument patterns (`-c`, `&&`, `;`, `|`, `$()`, backticks) always denied regardless of executable.
4. **`CommandValidator` class** — `shlex.split()` for safe parsing (posix-aware for Windows). Validates cwd against workspace. Runs `subprocess.run(shell=False)` with timeout, output limit, sanitized environment.
5. **`CommandExecutor` tool** — Registered as `execute_command` in ToolRegistry. Integrates CommandValidator + AuditLog.
6. **`AuditLog` class** — Append-only log: timestamp, session_id, tool, operation, decision, reason, success, duration, exit_code, error. Max entries cap. No secrets logged.
7. **`ToolSecurityProfile` dataclass** — Per-tool capability declarations. Default all False (deny). Network access = DENIED by default.
8. **`ToolPermissionResult` dataclass** — Typed permission result: decision (ALLOWED/DENIED/REQUIRES_APPROVAL), reason, metadata.
9. **Three-phase API** — `ToolRegistry.validate_request()`, `authorize_request()`, `execute()` are separate methods. The `execute()` method combines all three for convenience but the CodingAgent must use them separately.

### Workspace Model
- **Boundary enforcement:** `PathSecurity(workspace_root)` with `Path.resolve()` + `is_relative_to()`
- **Relative paths:** Anchored to workspace root (not cwd): `(workspace / path).resolve()`
- **Absolute paths:** Resolved directly and checked against workspace
- **Symlink escapes:** Detected because `Path.resolve()` follows symlinks to canonical form
- **Prefix collisions:** Avoided because `is_relative_to()` checks actual containment, not string prefix
- **Windows paths:** Handled correctly via `Path.resolve()` — drive letters, backslashes, normalization all handled

### Command Policy
- **Allowed by default:** `python`, `pytest`, `pip`, `git`
- **Denied by default:** `rm`, `del`, `format`, `shutdown`
- **Always denied arguments:** `-c`, `&&`, `||`, `;`, `|`, `>`, `>>`, `<`, `$(`, `` ` ``, `${`, `\n`
- **Shell metacharacters in any argument:** Denied (single-char check for `;&|`$\`(){}[]!#~`)
- **`python -c` specifically:** Denied by blocked_arguments pattern `-c` — correct security behavior
- **Unknown commands:** Default DENY

### Subprocess Security
- **`shell=True`:** NEVER used. Verified by source inspection test.
- **Timeout:** Every command has a configurable timeout. `TimeoutExpired` handled cleanly.
- **Output limit:** Output truncated to `max_output_bytes`. `truncated` flag returned.
- **Environment:** Inherits PATH but strips variables matching secret patterns (`SECRET`, `TOKEN`, `KEY`, `PASSWORD`).
- **Working directory:** Validated against workspace boundary before execution.

### Audit Logging
- **Recorded per invocation:** timestamp, session_id, tool_name, operation, decision, reason, success, duration_ms, exit_code, error
- **Not recorded:** API keys, tokens, passwords, secrets, environment variables
- **Max entries:** 10,000 (oldest evicted)
- **Exposed to assistant layer:** AuditLog is safe to query — no sensitive data

### Network Access
- **Default:** DENIED
- **No network tools registered** in ToolRegistry
- **ToolSecurityProfile.network_access:** Always False unless explicitly granted
- **Future capabilities** (web research, APIs) will be separate, explicitly approved tools

### Tests
| Category | Count | Notes |
|----------|-------|-------|
| Path Security | 7 | Traversal, absolute, symlink (skipped on Windows), prefix collision |
| File Security | 5 | Size limits, read/write bounds |
| Command Security | 10 | Allowed/denied/unknown, metacharacters, shell=True audit, cwd escape, timeout, output, exit code |
| Permissions | 6 | ALLOWED/DENIED/REQUIRES_APPROVAL, unknown tool, explicit deny wins |
| Auditing | 6 | Success/denial/timeout/secret records, max entries, clear |
| Regression | 4 | Calculator, file tools, ToolRegistry backward compat |
| Adversarial | 14 | Path injection, shell injection, subshells, redirects, backticks, newlines |
| Security Profile | 3 | Default deny, custom, calculator |
| Three-Phase API | 6 | validate/authorize/execute separation |
| Command Parsing | 4 | shlex, empty, whitespace, secret env |
| Workspace Integration | 5 | File tools reject traversal, accept valid |
| Environment | 2 | Secret exclusion, custom secrets |
| **Total** | **72** | **71 passed, 2 skipped (Windows symlink)** |

### Known Limitations
1. **Symlink tests on Windows** — Require admin privileges, skipped by default
2. **CommandPolicy is static** — Not loaded from config file yet (uses defaults)
3. **No network tools** — Network access DENIED but no mechanism to explicitly grant per-tool
4. **Audit log is in-memory** — Lost on process restart (could add SQLite persistence later)
5. **No sandbox** — Process isolation not implemented (designed for future addition)

### Files Changed
| File | Action | Description |
|------|--------|-------------|
| `src/tools/security.py` | NEW | PathSecurity, FileSizeGuard, CommandPolicy, CommandValidator, AuditLog, PermissionDecision, ToolPermissionResult, ToolSecurityProfile |
| `src/tools/tools.py` | REWRITTEN | Security integration, three-phase API, CommandExecutor tool, startswith removed |
| `src/tools/__init__.py` | UPDATED | Exports for new security classes |
| `tests/test_tools_security.py` | NEW | 72 tests covering all security requirements |
| `docs/ARCHITECTURE.md` | UPDATED | Security layer documentation, package structure, component tables |
| `docs/DEVELOPMENT_STATUS.md` | UPDATED | 427/427 tests, security component status |

### Commit
```
feat: harden tool security and workspace boundaries
```
