# Vercel Integration

## Overview

Voxline AI Core integrates with Vercel for deployment workflows.

## Architecture

```
CodingAgent
    ↓
VercelService (permission enforcement)
    ↓
VercelClient (HTTP transport)
    ↓
Vercel REST API
```

## Authentication

Set `VERCEL_TOKEN` environment variable:
```bash
export VERCEL_TOKEN=your_vercel_token_here
```

## Configuration

```bash
VERCEL_ENABLED=true
VERCEL_TOKEN=vrcl_...
VERCEL_ALLOWED_PROJECTS=prj_abc123,prj_def456
REQUIRE_PRODUCTION_APPROVAL=true
```

## Deployment Types

### Preview Deployment
- Created automatically when configured
- No approval required
- Use for: PR previews, testing changes

### Production Deployment
- Requires explicit user authorization
- Use for: final releases only
- The agent will NOT deploy to production automatically

## Deployment Workflow

1. Identify Vercel project
2. Identify corresponding repository/branch
3. Create deployment (preview or production)
4. Monitor deployment status
5. Return deployment URL/status

## Security

- Tokens never exposed to LLM
- Production deployment always requires explicit approval
- Project allowlists/denylists supported
- Every deployment operation is audited
