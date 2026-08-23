# Voxline Business Agent — local MVP

This first MVP runs entirely on the computer where you start it. It does not
send requests to OpenAI, Gemini, Claude, or another hosted AI service.

## Start

```powershell
python -m uvicorn src.api.server:app --host 127.0.0.1 --port 8000
```

Then open `http://127.0.0.1:8000/docs` in a browser. The interactive page
allows you to add business knowledge, search it, and create reviewable plans.

To enable chat, first download the configured local model into
`models/Qwen2.5-0.5B-Instruct`. After that, `POST /chat` accepts a `message`
and returns a response from the locally loaded model. The model download is a
one-time setup step; normal chat requests stay on this computer.

## Main endpoints

- `GET /health` — verifies that the local service is running.
- `POST /knowledge` — saves a business fact in the local SQLite memory.
- `GET /knowledge/search?query=...` — searches locally stored knowledge.
- `POST /plans` — creates a four-step business plan for a goal.
- `POST /chat` — answers a question with the local language model.

Example plan request:

```json
{
  "goal": "Increase sales for my service",
  "context": "We sell websites to small Armenian businesses."
}
```

The database appears as `memory/voxline_business.db` only after the local
service starts. Do not publish that file: it may contain business knowledge.

## Next development step

The current planner is deliberately deterministic and transparent. The next
phase is to connect it to a selected locally hosted language model, while
keeping the API and the business data on your machine.
