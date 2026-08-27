"""Build a real local git repository for end-to-end validation scenarios.

Creates:
  <base>/remote.git          — bare repo acting as the remote ("origin")
  <base>/repos/demo          — a working clone with an intentional bug + tests

The seeded `app.add()` returns 0 (wrong), and `app.greet()` is missing so the
tests in `tests/test_app.py` correctly FAIL before the fix and PASS after.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def run(cmd, cwd=None):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, shell=False)


def build_demo_repo(base: Path) -> Path:
    """Create the full demo repo layout. Returns path to working clone."""
    base = Path(base)
    remote = base / "remote.git"
    if remote.exists():
        shutil.rmtree(remote)
    r = run(["git", "init", "--bare", str(remote)])
    if r.returncode != 0:
        raise RuntimeError(r.stderr)

    clone = base / "repos" / "demo"
    if clone.exists():
        shutil.rmtree(clone)
    clone.mkdir(parents=True, exist_ok=True)

    run(["git", "init", "-b", "main", str(clone)])
    run(["git", "-C", str(clone), "config", "user.email", "voxline@example.com"])
    run(["git", "-C", str(clone), "config", "user.name", "Voxline Validation"])
    run(["git", "-C", str(clone), "config", "commit.gpgsign", "false"])

    (clone / "app.py").write_text(
        "def add(a, b):\n"
        "    return 0\n\n"
        "def bad_input():\n"
        "    return 0\n\n"
        "def main():\n"
        "    print('hello from app')\n\n"
        "if __name__ == '__main__':\n"
        "    main()\n",
        encoding="utf-8",
    )

    (clone / "main.py").write_text(
        "from app import greet\n\n"
        "if __name__ == '__main__':\n"
        "    print(greet('Voxline'))\n",
        encoding="utf-8",
    )

    (clone / "tests").mkdir()
    (clone / "tests" / "test_app.py").write_text(
        "import sys, os\n"
        "sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))\n"
        "import app\n\n"
        "def test_add():\n"
        "    assert app.add(2, 2) == 4\n"
        "    assert app.add(10, 5) == 15\n"
        "    assert app.add(-1, 1) == 0\n"
        "\n"
        "def test_greet():\n"
        "    assert app.greet('Ada') == 'Hello Ada'\n"
        "    assert app.greet('Voxline') == 'Hello Voxline'\n",
        encoding="utf-8",
    )

    (clone / "README.md").write_text(
        "# Demo Project\n\n"
        "A small Python utility library.\n\n"
        "## App\n\n"
        "- `app.add(a, b)` — returns the sum of two numbers.\n"
        "- `app.bad_input()` — returns 0.\n"
        "- `app.greet(name)` — returns `Hello <name>`.\n",
        encoding="utf-8",
    )

    run(["git", "-C", str(clone), "add", "-A"])
    run(["git", "-C", str(clone), "commit", "-m", "Initial demo repo (with intentional add() bug)"])
    run(["git", "-C", str(clone), "remote", "add", "origin", str(remote)])
    run(["git", "-C", str(clone), "push", "-u", "origin", "main"])
    # Ensure the bare remote's HEAD symref points at main so clones can checkout
    run(["git", "--git-dir", str(remote), "symbolic-ref", "HEAD", "refs/heads/main"])
    return clone
