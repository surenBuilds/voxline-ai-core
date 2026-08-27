# Browser Deployability (Phase 15): Vercel Serverless Deployment

This document describes the **deployment-ready** architecture added in Phase 15
to make the existing Voxline Coding Agent testable in a browser via Vercel.

The code is ready. **It has NOT been deployed/verified on Vercel from this
machine** because there is no `vercel` CLI / Vercel auth / GH_TOKEN here.
Follow the manual steps below to deploy; the exact routing and build settings
must be confirmed in your real Vercel project.

## What was added

| File | Purpose |
|------|---------|
| `src/providers/hosted.py` | `OpenAICompatProvider(AIProvider)` — calls any OpenAI-compatible `/chat/completions` API via `httpx` (async). Uses ONLY `AI_API_KEY`/`AI_BASE_URL`/`AI_MODEL` from env. Never logs/exposes the key. Header secret-safe errors. |
| `src/providers/factory.py` | Registers provider id `openai` (keeps `qwen`/`native`). |
| `src/config/settings.py` | New defaults+accessors `ai_api_key`, `ai_base_url`, `ai_model`; `AI_API_KEY` added to `SECRET_KEYS` (redaction). |
| `.env.example` | Placeholders only for hosted vars; corrected the stale `AI_PROVIDER=local_hf` default to `qwen`. |
| `api/index.py` | Lean Vercel ASGI FastAPI gateway. Reuses the real `ChatAssistant` / `BusinessAssistant` / `CodingAgent` / `ToolRegistry` / `MemoryStore` / security. Endpoints: `GET /` (existing UI), `GET /health`, `GET /api/tools`, `POST /api/chat`, `POST /api/business`, `POST /api/coding`. Generic errors only — never leaks secrets or tracebacks. |
| `vercel.json` | Serverless function config (`maxDuration` 120, `python3.12`), installs **lean** deps, rewrites all routes to the function. |
| `requirements-serverless.txt` | Lean install for the function (fastapi/uvicorn/httpx/pydantic/dotenv/loguru). Deliberately excludes torch/transformers so the build stays under serverless size limits. |
| `tests/test_hosted_provider.py` | Hosted-provider tests (request shape, response parsing, missing-key safety, no-secret leaks). |
| `tests/test_vercel_gateway.py` | Gateway tests (health/tools/chat/business/coding contracts; internal-error is generic and secret-safe). |

## Architecture

```
Browser
  -> Vercel frontend (gateway serves src/api/static/index.html at / ; relative endpoints only)
  -> api/index.py (server-side)
  -> OpenAICompatProvider -> hosted AI provider (AI_BASE_URL /chat/completions, Bearer AI_API_KEY)
  -> existing ChatAssistant / BusinessAssistant / CodingAgent / ToolRegistry / security
```

The host provider key is **server-side only** (Vercel env var). The browser
client never sees it; the client JS only calls relative endpoints and contains
no secrets.

## Manual deployment steps (must be done by the developer / CI with credentials)

1. **Prereqs:** a Vercel account, the `vercel` CLI (`npm i -g vercel`), and a
   hosted OpenAI-compatible endpoint + key. (None of these are available on
   this development machine.)
   ```
   vercel login
   vercel link
   ```

2. **Set server-side env vars in the Vercel project dashboard (or `vercel env add`):**
   | Variable | Requirement | Notes |
   |----------|-------------|-------|
   | `AI_PROVIDER` | yes | must be `openai` on Vercel (local `qwen`/`native` are NOT supported on serverless) |
   | `AI_API_KEY` | yes | **secret**; set server-side only, never in browser or repo |
   | `AI_BASE_URL` | yes | e.g. `https://api.openai.com/v1` (or your vLLM/Ollama-compatible gateway) |
   | `AI_MODEL` | yes | e.g. `gpt-3.5-turbo` |

3. **Deploy:**
   ```
   vercel deploy --prod
   ```
   `vercel.json` installs `requirements-serverless.txt` (lean) instead of the
   heavy root `requirements.txt`, keeping the function under size limits.

4. **Verify** `/health` returns `"status": "healthy"` and `GET /` shows the UI.

### Notes / caveats (be honest about these)

- **Routing:** with a single `api/index.py` ASGI function, the `vercel.json`
  `rewrites` sends `/`, `/health`, `/api/*` to the function (the FastAPI app
  dispatches each path). Confirm the rewrite preserves the original path in
  your actual Vercel project; adjust `vercel.json` if your project routes
  differently (e.g. a Next.js frontend that proxies `/api/*`).
- **Persistence is ephemeral:** on Vercel the filesystem (`/tmp`) is
  read-write only during a request and is lost between cold starts. The memory
  DB and Coding Agent workspace are ephemeral here. This is safe for a first
  interactive browser test but NOT durable state.
- **`/api/coding` limits:** full git clone + commit + GitHub PR + Vercel preview
  deployment is NOT reliable on an ephemeral serverless function (no `git`/
  `pytest` guarantee, function duration limits). Use `/api/coding` for simple
  in-request workspace tasks. For the full autonomous PR/deploy workflow you
  need a persistent worker (separate step, not part of this change).
- **Not usable on Vercel:** the local providers (`qwen`, `native`) and the
  local `serve_v04.py` web server. This gateway exists in parallel and does not
  replace them; local development still uses `qwen`/`native`.

## Local verification (done here)

- `python -m pytest tests/ --ignore=tests/test_providers.py --ignore=tests/test_real_e2e_validation.py` → **681 passed**.
  (The two ignored files are a pre-existing environment dependency on the local
  Qwen model, which is empty on this machine and unrelated to this change.)
- `tests/test_hosted_provider.py` and `tests/test_vercel_gateway.py` → **19 passed**.
- `git diff --check` → clean; no secrets in the diff.
