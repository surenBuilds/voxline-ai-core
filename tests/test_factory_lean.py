"""
Regression tests for the lean (serverless/Vercel) factory path.

The hosted ``openai`` provider must initialize without torch/transformers.
These tests run the factory inside a fresh subprocess with torch/transformers
blocked via ``sys.modules`` (mirroring the lean serverless bundle where those
packages are intentionally not installed), so they are isolated from any
already-registered providers in the parent process.

No real network / API key required: the openai provider is constructed and its
identity, error-safety and available-providers listing are asserted without
calling the hosted API.
"""

import os
import subprocess
import sys
import textwrap

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_LEAN_SCRIPT = textwrap.dedent(
    """
    import sys

    # Block the heavy dependencies, mirroring the lean serverless bundle.
    for _heavy in ("torch", "transformers", "tqdm"):
        sys.modules[_heavy] = None

    # 1. Importing the factory must succeed without torch/transformers.
    from src.providers import factory
    from src.providers.factory import ProviderFactory, _ensure_builtin_providers

    # 2. Registering/ensuring built-ins must not crash in the lean env.
    _ensure_builtin_providers()

    # 3. get_available_providers() must not force/require torch. Heavy modules
    #    must remain the None sentinel (i.e. never actually imported).
    available = ProviderFactory.get_available_providers()
    assert "openai" in available, available
    for _h in ("torch", "transformers", "tqdm"):
        assert sys.modules.get(_h) is None, (
            f"{_h} must not be imported merely to list providers"
        )
    print("AVAILABLE:", sorted(available))
    print("CHECK_AVAILABLE_NO_TORCH:OK")

    # 4. Creating the openai provider must succeed.
    from src.config.settings import VoxlineConfig
    vc = VoxlineConfig.__new__(VoxlineConfig)
    vc._config = {
        "AI_PROVIDER": "openai",
        "AI_DEVICE": "cpu",
        "AI_API_KEY": "",
        "AI_BASE_URL": "https://api.openai.com/v1",
        "AI_MODEL": "gpt-3.5-turbo",
    }
    provider = ProviderFactory.create(vc)
    assert provider.provider_id == "openai", provider.provider_id
    print("CREATE_OPENAI:OK")
    print("OPENAI_TYPE:", type(provider).__name__)

    # 5. No heavy module may have been actually imported during the above.
    for _h in ("torch", "transformers", "tqdm"):
        assert sys.modules.get(_h) is None, (
            f"{_h} must not be imported during openai provider creation"
        )
    print("NO_TORCH_IMPORT:OK")
    """
)


def _run_lean() -> tuple:
    """Run the lean-env subprocess and return (returncode, stdout, stderr)."""
    proc = subprocess.run(
        [sys.executable, "-c", _LEAN_SCRIPT],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return proc.returncode, proc.stdout, proc.stderr


def test_lean_env_openai_initializes_without_torch():
    code, out, err = _run_lean()
    assert code == 0, f"lean subprocess failed.\nSTDOUT:\n{out}\nSTDERR:\n{err}"
    assert "AVAILABLE:" in out
    assert "CHECK_AVAILABLE_NO_TORCH:OK" in out
    assert "CREATE_OPENAI:OK" in out
    assert "OPENAI_TYPE: OpenAICompatProvider" in out, out
    assert "NO_TORCH_IMPORT:OK" in out


def test_lean_env_get_available_providers_does_not_force_torch():
    code, out, err = _run_lean()
    assert code == 0, f"lean subprocess failed.\nSTDOUT:\n{out}\nSTDERR:\n{err}"
    available_line = next(
        (l for l in out.splitlines() if l.startswith("AVAILABLE:")), ""
    )
    assert "openai" in available_line
    # openai must be listed; native/qwen must NOT be auto-registered when torch
    # is absent (they would otherwise force a heavy import).
    assert "native" not in available_line, available_line
    assert "qwen" not in available_line, available_line


def test_factory_imports_in_lean_env():
    code, out, err = _run_lean()
    assert code == 0, f"lean subprocess failed.\nSTDOUT:\n{out}\nSTDERR:\n{err}"
    assert out.strip(), "factory import produced no output"


def test_openai_provider_identity_without_torch():
    """Direct in-process smoke check that openai needs only httpx, not torch."""
    from src.providers.hosted import OpenAICompatProvider
    from src.config.settings import VoxlineConfig

    vc = VoxlineConfig.__new__(VoxlineConfig)
    vc._config = {
        "AI_PROVIDER": "openai",
        "AI_DEVICE": "cpu",
        "AI_API_KEY": "",
        "AI_BASE_URL": "https://api.openai.com/v1",
        "AI_MODEL": "gpt-3.5-turbo",
    }
    provider = OpenAICompatProvider(config=vc, api_key="")
    assert provider.provider_id == "openai"

    from src.providers.factory import ProviderFactory
    created = ProviderFactory.create(vc)
    assert created.provider_id == "openai"
