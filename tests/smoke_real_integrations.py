"""
Optional real integration smoke tests (Phase 7 Step 12).

Gated behind VOXLINE_EXTERNAL_SMOKE=1 environment variable.
Without it, all tests are skipped.

With it:
  - Verify GitHub authentication
  - Verify repository access
  - Verify Vercel authentication
  - Verify project access

NEVER creates production deployments.
NEVER merges PRs.
NEVER deletes anything.
"""

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@unittest.skipUnless(
    os.environ.get("VOXLINE_EXTERNAL_SMOKE"),
    "External smoke tests disabled (set VOXLINE_EXTERNAL_SMOKE=1 to enable)",
)
class TestGitHubSmoke(unittest.TestCase):
    """Real GitHub API smoke tests."""

    def setUp(self):
        from src.integrations.credentials import EnvironmentCredentialProvider
        self.creds = EnvironmentCredentialProvider()
        if not self.creds.is_available("github"):
            self.skipTest("GITHUB_TOKEN not set")

    def test_01_github_auth(self):
        """Verify GitHub token is valid."""
        from src.integrations.github.client import GitHubClient
        token = self.creds.get_token("github")
        client = GitHubClient(token)
        repos = client.list_repositories(per_page=1)
        self.assertIsInstance(repos, list)

    def test_02_github_repo_access(self):
        """Verify we can read a configured repo."""
        from src.integrations.github.client import GitHubClient
        token = self.creds.get_token("github")
        client = GitHubClient(token)
        repos = client.list_repositories(per_page=1)
        if repos:
            repo = repos[0]
            self.assertTrue(len(repo.owner) > 0)
            self.assertTrue(len(repo.name) > 0)


@unittest.skipUnless(
    os.environ.get("VOXLINE_EXTERNAL_SMOKE"),
    "External smoke tests disabled (set VOXLINE_EXTERNAL_SMOKE=1 to enable)",
)
class TestVercelSmoke(unittest.TestCase):
    """Real Vercel API smoke tests."""

    def setUp(self):
        from src.integrations.credentials import EnvironmentCredentialProvider
        self.creds = EnvironmentCredentialProvider()
        if not self.creds.is_available("vercel"):
            self.skipTest("VERCEL_TOKEN not set")

    def test_03_vercel_auth(self):
        """Verify Vercel token is valid."""
        from src.integrations.vercel.client import VercelClient
        token = self.creds.get_token("vercel")
        client = VercelClient(token)
        projects = client.list_projects(per_page=1)
        self.assertIsInstance(projects, list)

    def test_04_vercel_project_access(self):
        """Verify we can list projects."""
        from src.integrations.vercel.client import VercelClient
        token = self.creds.get_token("vercel")
        client = VercelClient(token)
        projects = client.list_projects(per_page=1)
        if projects:
            project = projects[0]
            self.assertTrue(len(project.id) > 0)
            self.assertTrue(len(project.name) > 0)


if __name__ == "__main__":
    unittest.main()
