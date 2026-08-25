# GitHub Integration

## Overview

Voxline AI Core integrates with GitHub through a typed client/service/security stack.

## Architecture

```
CodingAgent
    ↓
GitHubService (permission enforcement)
    ↓
GitHubClient (HTTP transport)
    ↓
GitHub REST API
```

## Authentication

Tokens are obtained through `CredentialProvider`, never directly from environment variables by the agent.

Set `GITHUB_TOKEN` environment variable:
```bash
export GITHUB_TOKEN=ghp_your_token_here
```

## Configuration

```bash
GITHUB_ENABLED=true
GITHUB_TOKEN=ghp_...
GITHUB_ALLOWED_REPOSITORIES=owner/repo1,owner/repo2
```

## Permission Policy

| Operation | Default | Requires Approval |
|-----------|---------|-------------------|
| READ (metadata, files, commits, issues, PRs) | Allowed | No |
| WRITE (branch, commit, PR, issue) | Allowed | Configurable |
| DESTRUCTIVE (delete branch, merge PR, force push) | Denied | Always |

## Repository Workflow

1. Identify repository
2. Inspect repository metadata
3. Create feature branch
4. Clone/prepare workspace
5. Analyze code
6. Modify files
7. Run tests
8. Inspect git diff
9. Commit changes
10. Push branch
11. Create pull request
12. Return PR information

## Security

- Tokens never exposed to LLM context
- Tokens never logged or returned in API responses
- All operations classified as READ/WRITE/DESTRUCTIVE
- Destructive operations require explicit user approval
- Repository allowlists/denylists supported
