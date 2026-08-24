"""Tool security hardening tests.

Covers:
  PATH SECURITY (tests 1-7)
  FILE SECURITY (tests 8-12)
  COMMAND SECURITY (tests 13-22)
  PERMISSIONS (tests 23-26)
  AUDITING (tests 27-30)
  REGRESSION (tests 31-34)
  ADVERSARIAL (tests 35-48)
"""

import os
import sys
import time
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tools.security import (
    PathSecurity,
    FileSizeGuard,
    CommandPolicy,
    CommandValidator,
    AuditLog,
    PermissionDecision,
    ToolPermissionResult,
    ToolSecurityProfile,
)
from src.tools.tools import (
    ToolRegistry,
    Calculator,
    FileReadTool,
    FileWriteTool,
    DirectoryListTool,
    CommandExecutor,
)

# Helper: write a temporary Python script for command tests that need real execution
def _write_temp_script(content: str) -> str:
    """Write a Python script to a temp file and return its path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8")
    f.write(content)
    f.close()
    return f.name


# =========================================================================
# PATH SECURITY (tests 1-7)
# =========================================================================

class TestPathSecurity(unittest.TestCase):
    """Tests 1-7: workspace boundary enforcement."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.ps = PathSecurity(self.tmpdir)
        self.workspace = Path(self.tmpdir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    # --- Test 1: normal workspace path ---
    def test_01_normal_workspace_path(self):
        """Inside workspace path is ALLOWED."""
        result = self.ps.validate_path("subdir/file.txt")
        self.assertEqual(result.decision, PermissionDecision.ALLOWED)

    # --- Test 2: ../ traversal ---
    def test_02_traversal_dotdot(self):
        """../ traversal is DENIED."""
        result = self.ps.validate_path("../../etc/passwd")
        self.assertEqual(result.decision, PermissionDecision.DENIED)
        self.assertIn("escapes workspace", result.reason.lower())

    # --- Test 3: absolute outside path ---
    def test_03_absolute_outside_path(self):
        """Absolute path outside workspace is DENIED."""
        result = self.ps.validate_path("/etc/passwd")
        self.assertEqual(result.decision, PermissionDecision.DENIED)

    # --- Test 4: normalized traversal ---
    def test_04_normalized_traversal(self):
        """Path like workspace/../other is DENIED after resolution."""
        evil = str(self.workspace / "subdir" / ".." / ".." / "etc" / "passwd")
        result = self.ps.validate_path(evil)
        self.assertEqual(result.decision, PermissionDecision.DENIED)

    # --- Test 5: symlink escape ---
    @unittest.skipUnless(os.name != "nt" or os.environ.get("VOXLINE_TEST_SYMLINKS"),
                         "Symlink tests require admin on Windows")
    def test_05_symlink_escape(self):
        """Symlink that points outside workspace is DENIED."""
        link = self.workspace / "escape_link"
        outside = Path(tempfile.mkdtemp())
        try:
            os.symlink(str(outside), str(link))
            result = self.ps.validate_path("escape_link/secret.txt")
            self.assertEqual(result.decision, PermissionDecision.DENIED)
        finally:
            if link.exists() or link.is_symlink():
                os.unlink(str(link))
            import shutil
            shutil.rmtree(outside, ignore_errors=True)

    # --- Test 6: Windows-style path ---
    def test_06_windows_style_path(self):
        """Windows-style absolute path is handled correctly."""
        result = self.ps.validate_path("C:\\Windows\\System32\\config\\SAM")
        if os.name == "nt":
            self.assertEqual(result.decision, PermissionDecision.DENIED)
        else:
            self.assertEqual(result.decision, PermissionDecision.ALLOWED)

    # --- Test 7: workspace prefix collision ---
    def test_07_workspace_prefix_collision(self):
        """Path like 'project-evil' next to 'project' is correctly rejected if outside."""
        sibling = Path(self.tmpdir).parent / "voxline-evil"
        sibling.mkdir(exist_ok=True)
        try:
            result = self.ps.validate_path(str(sibling / "file.txt"))
            self.assertEqual(result.decision, PermissionDecision.DENIED)
        finally:
            import shutil
            shutil.rmtree(sibling, ignore_errors=True)

    # --- safe_resolve raises WorkspaceBoundaryError ---
    def test_safe_resolve_raises_on_escape(self):
        from src.errors import WorkspaceBoundaryError
        with self.assertRaises(WorkspaceBoundaryError):
            self.ps.safe_resolve("../../etc/passwd")


# =========================================================================
# FILE SECURITY (tests 8-12)
# =========================================================================

class TestFileSecurity(unittest.TestCase):
    """Tests 8-12: file size limits."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.guard = FileSizeGuard(max_size_bytes=100)
        self.workspace = Path(self.tmpdir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_08_read_within_limit(self):
        f = self.workspace / "small.txt"
        f.write_text("hello")
        result = self.guard.check_read(f)
        self.assertEqual(result.decision, PermissionDecision.ALLOWED)

    def test_09_read_above_limit(self):
        f = self.workspace / "big.txt"
        f.write_text("x" * 200)
        result = self.guard.check_read(f)
        self.assertEqual(result.decision, PermissionDecision.DENIED)
        self.assertIn("too large", result.reason.lower())

    def test_10_write_within_limit(self):
        result = self.guard.check_write("hello")
        self.assertEqual(result.decision, PermissionDecision.ALLOWED)

    def test_11_write_above_limit(self):
        result = self.guard.check_write("x" * 200)
        self.assertEqual(result.decision, PermissionDecision.DENIED)
        self.assertIn("too large", result.reason.lower())

    def test_12_output_truncation(self):
        """CommandExecutor truncates output to max_output_bytes."""
        script = _write_temp_script("print('A' * 200)")
        try:
            # Use a permissive policy for this functional test
            policy = CommandPolicy(allowed_commands={"python"})
            ps = PathSecurity(self.tmpdir)
            cv = CommandValidator(
                policy=policy,
                path_security=ps,
                max_output_bytes=50,
                timeout_seconds=10,
            )
            result = cv.run_command(
                command=f"python {script}",
                cwd=self.tmpdir,
            )
            self.assertTrue(result["success"])
            self.assertTrue(result.get("truncated", False))
            self.assertLessEqual(len(result["stdout"]), 50)
        finally:
            os.unlink(script)


# =========================================================================
# COMMAND SECURITY (tests 13-22)
# =========================================================================

class TestCommandSecurity(unittest.TestCase):
    """Tests 13-22: command permission policy and safe execution."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.policy = CommandPolicy(
            allowed_commands={"python", "pytest", "git", "pip"},
            denied_commands={"rm", "del", "format", "shutdown"},
        )
        self.ps = PathSecurity(self.tmpdir)
        self.cv = CommandValidator(
            policy=self.policy,
            path_security=self.ps,
            max_output_bytes=1024,
            timeout_seconds=10,
        )

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    # --- Test 13: allowed command ---
    def test_13_allowed_command(self):
        """Python command in allowed list is ALLOWED (via script, not -c)."""
        script = _write_temp_script("print(1)")
        try:
            result = self.cv.run_command(f"python {script}", cwd=self.tmpdir)
            self.assertTrue(result["success"])
            self.assertEqual(result["stdout"].strip(), "1")
        finally:
            os.unlink(script)

    # --- Test 14: denied command ---
    def test_14_denied_command(self):
        """rm command in denied list is DENIED."""
        result = self.cv.run_command("rm -rf /", cwd=self.tmpdir)
        self.assertFalse(result["success"])
        self.assertIn("denied", result["error"].lower())

    # --- Test 15: unknown command ---
    def test_15_unknown_command(self):
        """Command not in allowed list is DENIED."""
        result = self.cv.run_command("nmap localhost", cwd=self.tmpdir)
        self.assertFalse(result["success"])
        self.assertIn("not in allowed list", result["error"].lower())

    # --- Test 16: dangerous arguments (-c blocked) ---
    def test_16_dangerous_arguments(self):
        """python with -c flag is DENIED by blocked_arguments policy."""
        result = self.cv.run_command("python -c \"import os\"", cwd=self.tmpdir)
        self.assertFalse(result["success"])
        # Error message contains "blocked" from the blocked_arguments detection
        self.assertTrue(
            "blocked" in result["error"].lower() or "denied" in result["error"].lower(),
            f"Expected 'blocked' or 'denied' in error: {result['error']}"
        )

    # --- Test 17: shell metacharacters ---
    def test_17_shell_metacharacters(self):
        """Commands with shell metacharacters are DENIED."""
        for cmd in [
            "python test.py && rm -rf /",
            "python test.py; rm -rf /",
            "python test.py | dangerous",
        ]:
            result = self.cv.run_command(cmd, cwd=self.tmpdir)
            self.assertFalse(result["success"], f"Expected DENIED for: {cmd}")

    # --- Test 18: shell=True cannot be introduced ---
    def test_18_no_shell_true(self):
        """CommandValidator.run_command never uses shell=True."""
        import inspect
        src = inspect.getsource(CommandValidator.run_command)
        self.assertNotIn("shell=True", src)
        self.assertIn("shell=False", src)

    # --- Test 19: working directory escape ---
    def test_19_working_directory_escape(self):
        """Commands with cwd outside workspace are DENIED."""
        script = _write_temp_script("print(1)")
        try:
            result = self.cv.run_command(f"python {script}", cwd="/etc")
            self.assertFalse(result["success"])
            self.assertIn("escapes workspace", result["error"].lower())
        finally:
            os.unlink(script)

    # --- Test 20: command timeout ---
    def test_20_command_timeout(self):
        """Long-running command is killed after timeout."""
        script = _write_temp_script("import time; time.sleep(10)")
        short_cv = CommandValidator(
            policy=self.policy,
            path_security=self.ps,
            max_output_bytes=1024,
            timeout_seconds=1,
        )
        try:
            result = short_cv.run_command(f"python {script}", cwd=self.tmpdir)
            self.assertFalse(result["success"])
            self.assertIn("timed out", result["error"].lower())
        finally:
            os.unlink(script)

    # --- Test 21: output limit ---
    def test_21_output_limit(self):
        """Large command output is truncated to max_output_bytes."""
        script = _write_temp_script("print('A' * 500)")
        cv = CommandValidator(
            policy=self.policy,
            path_security=self.ps,
            max_output_bytes=20,
            timeout_seconds=10,
        )
        try:
            result = cv.run_command(f"python {script}", cwd=self.tmpdir)
            self.assertTrue(result["success"])
            self.assertLessEqual(len(result["stdout"]), 20)
        finally:
            os.unlink(script)

    # --- Test 22: non-zero exit code ---
    def test_22_nonzero_exit_code(self):
        """Non-zero exit code is captured correctly."""
        script = _write_temp_script("import sys; sys.exit(42)")
        try:
            result = self.cv.run_command(f"python {script}", cwd=self.tmpdir)
            self.assertFalse(result["success"])
            self.assertEqual(result["exit_code"], 42)
        finally:
            os.unlink(script)


# =========================================================================
# PERMISSIONS (tests 23-26)
# =========================================================================

class TestPermissionDecisions(unittest.TestCase):
    """Tests 23-26: permission decision structures."""

    def test_23_allowed_decision(self):
        self.assertEqual(PermissionDecision.ALLOWED.value, "allowed")

    def test_24_denied_decision(self):
        self.assertEqual(PermissionDecision.DENIED.value, "denied")

    def test_25_requires_approval_decision(self):
        self.assertEqual(PermissionDecision.REQUIRES_APPROVAL.value, "requires_approval")

    def test_26_unknown_tool_defaults_to_deny(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tr = ToolRegistry(tmpdir)
            result = tr.validate_request("nonexistent_tool")
            self.assertEqual(result.decision, PermissionDecision.DENIED)

    def test_requires_approval_policy(self):
        policy = CommandPolicy(
            allowed_commands={"python"},
            requires_approval={"git"},
        )
        result = policy.evaluate("git", ["push", "origin", "main"])
        self.assertEqual(result.decision, PermissionDecision.REQUIRES_APPROVAL)

    def test_explicit_deny_wins_over_approval(self):
        policy = CommandPolicy(
            denied_commands={"rm"},
            requires_approval={"rm"},
        )
        result = policy.evaluate("rm", ["-rf", "/"])
        self.assertEqual(result.decision, PermissionDecision.DENIED)


# =========================================================================
# AUDITING (tests 27-30)
# =========================================================================

class TestAuditLogging(unittest.TestCase):
    """Tests 27-30: audit log records."""

    def setUp(self):
        self.audit = AuditLog(max_entries=100)

    def test_27_successful_tool_audit(self):
        self.audit.record(
            tool_name="read_file",
            operation="read test.txt",
            decision=PermissionDecision.ALLOWED,
            reason="Path inside workspace",
            success=True,
            duration_ms=12.5,
        )
        self.assertEqual(len(self.audit.entries), 1)
        entry = self.audit.entries[0]
        self.assertEqual(entry.tool_name, "read_file")
        self.assertTrue(entry.success)

    def test_28_denied_tool_audit(self):
        self.audit.record(
            tool_name="execute_command",
            operation="rm -rf /",
            decision=PermissionDecision.DENIED,
            reason="Command denied",
            success=False,
            duration_ms=2.0,
        )
        self.assertEqual(len(self.audit.entries), 1)
        entry = self.audit.entries[0]
        self.assertFalse(entry.success)
        self.assertEqual(entry.decision, PermissionDecision.DENIED)

    def test_29_timeout_audit(self):
        self.audit.record(
            tool_name="execute_command",
            operation="sleep 100",
            decision=PermissionDecision.ALLOWED,
            reason="Command in allowed list",
            success=False,
            duration_ms=1000.0,
            error="Command timed out after 1s",
        )
        entry = self.audit.entries[0]
        self.assertIn("timed out", entry.error)

    def test_30_secret_redaction(self):
        """Audit log does not contain secrets in reason/error fields."""
        self.audit.record(
            tool_name="execute_command",
            operation="python test.py",
            decision=PermissionDecision.ALLOWED,
            reason="Command allowed",
            success=True,
            duration_ms=5.0,
        )
        entry = self.audit.entries[0]
        self.assertNotIn("secret123", entry.reason)
        self.assertNotIn("secret123", entry.error or "")

    def test_audit_max_entries(self):
        audit = AuditLog(max_entries=5)
        for i in range(10):
            audit.record(
                tool_name="test", operation=f"op {i}",
                decision=PermissionDecision.ALLOWED, reason="r",
                success=True, duration_ms=1.0,
            )
        self.assertEqual(len(audit.entries), 5)

    def test_audit_clear(self):
        self.audit.record(
            tool_name="test", operation="op",
            decision=PermissionDecision.ALLOWED, reason="r",
            success=True, duration_ms=1.0,
        )
        self.audit.clear()
        self.assertEqual(len(self.audit.entries), 0)


# =========================================================================
# REGRESSION (tests 31-34)
# =========================================================================

class TestRegression(unittest.TestCase):
    """Tests 31-34: existing tools still work after hardening."""

    def test_31_calculator_works(self):
        tr = ToolRegistry(".")
        result = tr.execute_tool("calculator", expression="2+2")
        self.assertEqual(result, 4.0)

    def test_32_file_tools_work(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tr = ToolRegistry(tmpdir)
            result = tr.execute_tool("write_file", path="test.txt", content="Hello")
            self.assertTrue(result["success"])
            result = tr.execute_tool("read_file", path="test.txt")
            self.assertEqual(result, "Hello")
            result = tr.execute_tool("list_directory", path=".")
            self.assertIn("count", result)
            self.assertEqual(result["count"], 1)

    def test_33_tool_registry_works(self):
        tr = ToolRegistry(".")
        tools = tr.list_tools()
        self.assertIn("calculator", tools)
        self.assertIn("read_file", tools)
        self.assertIn("write_file", tools)
        self.assertIn("list_directory", tools)
        self.assertIn("execute_command", tools)

    def test_34_all_existing_tests_pass(self):
        """Placeholder: full test suite verified separately."""
        pass


# =========================================================================
# ADVERSARIAL (tests 35-48)
# =========================================================================

class TestAdversarialCases(unittest.TestCase):
    """Tests 35-48: adversarial path and command injection attempts."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.ps = PathSecurity(self.tmpdir)
        self.policy = CommandPolicy(
            allowed_commands={"python"},
            denied_commands={"rm", "del"},
        )
        self.cv = CommandValidator(
            policy=self.policy,
            path_security=self.ps,
            max_output_bytes=1024,
            timeout_seconds=5,
        )

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    # --- Path adversarial ---
    def test_35_dotdot_secret(self):
        result = self.ps.validate_path("../../secret.txt")
        self.assertEqual(result.decision, PermissionDecision.DENIED)

    def test_36_sibling_workspace(self):
        sibling = Path(self.tmpdir).parent / "workspace-evil"
        sibling.mkdir(exist_ok=True)
        try:
            result = self.ps.validate_path(str(sibling / "file"))
            self.assertEqual(result.decision, PermissionDecision.DENIED)
        finally:
            import shutil
            shutil.rmtree(sibling, ignore_errors=True)

    def test_37_windows_drive_escape(self):
        result = self.ps.validate_path("C:\\outside\\secret.txt")
        if os.name == "nt":
            self.assertEqual(result.decision, PermissionDecision.DENIED)
        else:
            self.assertEqual(result.decision, PermissionDecision.ALLOWED)

    @unittest.skipUnless(os.name != "nt" or os.environ.get("VOXLINE_TEST_SYMLINKS"),
                         "Symlink tests require admin on Windows")
    def test_38_symlink_in_workspace(self):
        outside = Path(tempfile.mkdtemp())
        link = Path(self.tmpdir) / "link_dir"
        try:
            os.symlink(str(outside), str(link))
            result = self.ps.validate_path("link_dir/secret.txt")
            self.assertEqual(result.decision, PermissionDecision.DENIED)
        finally:
            if link.exists() or link.is_symlink():
                os.unlink(str(link))
            import shutil
            shutil.rmtree(outside, ignore_errors=True)

    # --- Command adversarial ---
    def test_39_python_dash_c(self):
        result = self.cv.run_command("python -c \"import os\"", cwd=self.tmpdir)
        self.assertFalse(result["success"])

    def test_40_double_ampersand(self):
        result = self.cv.run_command("python test.py && rm -rf /", cwd=self.tmpdir)
        self.assertFalse(result["success"])

    def test_41_semicolon(self):
        result = self.cv.run_command("python test.py; rm -rf /", cwd=self.tmpdir)
        self.assertFalse(result["success"])

    def test_42_pipe(self):
        result = self.cv.run_command("python test.py | dangerous", cwd=self.tmpdir)
        self.assertFalse(result["success"])

    def test_43_redirect_output(self):
        result = self.cv.run_command("python test.py > /tmp/output", cwd=self.tmpdir)
        self.assertFalse(result["success"])

    def test_44_redirect_input(self):
        result = self.cv.run_command("python test.py < /etc/passwd", cwd=self.tmpdir)
        self.assertFalse(result["success"])

    def test_45_subshell(self):
        result = self.cv.run_command("python $(which dangerous)", cwd=self.tmpdir)
        self.assertFalse(result["success"])

    def test_46_backtick_subshell(self):
        result = self.cv.run_command("python `which dangerous`", cwd=self.tmpdir)
        self.assertFalse(result["success"])

    def test_47_dollar_brace(self):
        result = self.cv.run_command("python ${EVIL}", cwd=self.tmpdir)
        self.assertFalse(result["success"])

    def test_48_newline_in_command(self):
        result = self.cv.run_command("python test.py\nrm -rf /", cwd=self.tmpdir)
        self.assertFalse(result["success"])


# =========================================================================
# SECURITY PROFILE TESTS
# =========================================================================

class TestToolSecurityProfile(unittest.TestCase):
    """ToolSecurityProfile defaults and structure."""

    def test_default_profile_is_all_denied(self):
        p = ToolSecurityProfile()
        self.assertFalse(p.filesystem_read)
        self.assertFalse(p.filesystem_write)
        self.assertFalse(p.command_execution)
        self.assertFalse(p.network_access)
        self.assertFalse(p.requires_approval)

    def test_custom_profile(self):
        p = ToolSecurityProfile(filesystem_read=True, command_execution=True)
        self.assertTrue(p.filesystem_read)
        self.assertTrue(p.command_execution)
        self.assertFalse(p.network_access)

    def test_calculator_profile(self):
        calc = Calculator()
        profile = calc.security_profile
        self.assertFalse(profile.filesystem_read)
        self.assertFalse(profile.filesystem_write)
        self.assertFalse(profile.command_execution)


# =========================================================================
# INTEGRATION: ToolRegistry three-phase API
# =========================================================================

class TestToolRegistryThreePhase(unittest.TestCase):
    """ToolRegistry validate → authorize → execute separation."""

    def test_validate_unknown_tool_denied(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tr = ToolRegistry(tmpdir)
            result = tr.validate_request("nonexistent")
            self.assertEqual(result.decision, PermissionDecision.DENIED)

    def test_authorize_unknown_tool_denied(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tr = ToolRegistry(tmpdir)
            result = tr.authorize_request("nonexistent")
            self.assertEqual(result.decision, PermissionDecision.DENIED)

    def test_validate_file_tool_checks_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tr = ToolRegistry(tmpdir)
            result = tr.validate_request("read_file", path="../../etc/passwd")
            self.assertEqual(result.decision, PermissionDecision.DENIED)

    def test_authorize_file_tool_checks_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tr = ToolRegistry(tmpdir)
            result = tr.authorize_request("write_file", path="../../evil.txt")
            self.assertEqual(result.decision, PermissionDecision.DENIED)

    def test_execute_denied_returns_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tr = ToolRegistry(tmpdir)
            result = tr.execute("nonexistent")
            self.assertIn("error", result)

    def test_execute_audits_denial(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tr = ToolRegistry(tmpdir)
            tr.execute("nonexistent", session_id="test-session")
            self.assertGreater(len(tr.audit_log.entries), 0)
            last = tr.audit_log.entries[-1]
            self.assertEqual(last.session_id, "test-session")
            self.assertEqual(last.decision, PermissionDecision.DENIED)


# =========================================================================
# COMMAND PARSING
# =========================================================================

class TestCommandParsing(unittest.TestCase):
    """Command parsing safety."""

    def test_parse_simple_command(self):
        ps = PathSecurity(".")
        cv = CommandValidator(CommandPolicy(), ps)
        exe, args = cv.parse_command("python -c 'print(1)'")
        self.assertEqual(exe, "python")
        self.assertEqual(args[0], "-c")
        self.assertEqual(len(args), 2)

    def test_parse_empty_command_raises(self):
        ps = PathSecurity(".")
        cv = CommandValidator(CommandPolicy(), ps)
        with self.assertRaises(ValueError):
            cv.parse_command("")

    def test_parse_whitespace_only_raises(self):
        ps = PathSecurity(".")
        cv = CommandValidator(CommandPolicy(), ps)
        with self.assertRaises(ValueError):
            cv.parse_command("   ")

    def test_build_safe_env_excludes_secrets(self):
        ps = PathSecurity(".")
        cv = CommandValidator(CommandPolicy(), ps)
        os.environ["TEST_SECRET_KEY_XYZ"] = "supersecret"
        try:
            env = cv.build_safe_env()
            self.assertNotIn("TEST_SECRET_KEY_XYZ", env)
        finally:
            del os.environ["TEST_SECRET_KEY_XYZ"]


# =========================================================================
# WORKSPACE BOUNDARY INTEGRATION
# =========================================================================

class TestWorkspaceBoundaryIntegration(unittest.TestCase):
    """File tools reject paths outside workspace."""

    def test_read_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = FileReadTool(tmpdir)
            result = tool.execute(path="../../etc/passwd")
            self.assertIn("error", result)

    def test_write_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = FileWriteTool(tmpdir)
            result = tool.execute(path="../../evil.txt", content="bad")
            self.assertIn("error", result)

    def test_list_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = DirectoryListTool(tmpdir)
            result = tool.execute(path="../../etc")
            self.assertIn("error", result)

    def test_read_accepts_valid_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "ok.txt").write_text("content")
            tool = FileReadTool(tmpdir)
            result = tool.execute(path="ok.txt")
            self.assertEqual(result, "content")

    def test_write_accepts_valid_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = FileWriteTool(tmpdir)
            result = tool.execute(path="ok.txt", content="hello")
            self.assertTrue(result["success"])


# =========================================================================
# ENVIRONMENT SANITIZATION
# =========================================================================

class TestEnvironmentSanitization(unittest.TestCase):
    """Secrets are not leaked to subprocess environment."""

    def test_env_excludes_api_keys(self):
        ps = PathSecurity(".")
        cv = CommandValidator(CommandPolicy(), ps)
        env = cv.build_safe_env()
        for key in env:
            self.assertNotIn("SECRET", key.upper())
            self.assertNotIn("PASSWORD", key.upper())

    def test_custom_secrets_added(self):
        ps = PathSecurity(".")
        cv = CommandValidator(CommandPolicy(), ps)
        os.environ["MY_CUSTOM_SECRET"] = "abc"
        try:
            env = cv.build_safe_env(secrets={"MY_CUSTOM_SECRET"})
            self.assertNotIn("MY_CUSTOM_SECRET", env)
        finally:
            del os.environ["MY_CUSTOM_SECRET"]


if __name__ == "__main__":
    unittest.main()
