# Phase 7 — Step 3-4 Report

## Implementation Summary

### Step 3: ContextBuilder

Created `src/assistant/context.py` with a production-quality ContextBuilder responsible for assembling structured context that reaches the AIProvider.

**Key design decisions:**
- System instructions are NOT included in ContextBuilder output — the provider handles them via `chat()`. This avoids double-system-message issues with QwenProvider.
- Memory is retrieved from MemoryStore and formatted into a structured `[RELEVANT MEMORY]` block that is injected into the message list — fixing the architectural bug where memory was retrieved but never injected.
- Character budgets prevent unbounded prompt growth: 25% memory, 50% history, 25% overhead.
- Context sections are ordered: memory → mode instruction → history → business context → workspace context → user message.
- Graceful error handling: memory retrieval failures produce empty context (never crash).

**Context sections:**
| Section | Content | Budget |
|---------|---------|--------|
| MEMORY | Retrieved facts from MemoryStore | 25% |
| MODE INSTRUCTION | Mode-specific behavior guidance | 12.5% |
| CONVERSATION | Recent session history | 50% |
| BUSINESS_CONTEXT | Business domain context (business mode only) | 12.5% |
| WORKSPACE_CONTEXT | Workspace path (coding mode only) | 12.5% |
| USER_REQUEST | The user's current message | uncapped |

### Step 4: ChatAssistant

Created `src/assistant/chat.py` with ChatAssistant — the main conversational interface.

**Architecture compliance:**
- Uses AIProvider exclusively — no direct Qwen or TextGenerator usage.
- Provider is injectable and replaceable.
- All intelligence flows through: Assistant → ContextBuilder → AIProvider → Provider.

**Flow:**
```
User message
  → validate input
  → SessionManager.get(session_id)
  → ContextBuilder.build(session, message)
  → AIProvider.chat(messages, config)
  → Session.add_message(user)
  → Session.add_message(assistant)
  → Optional: MemoryStore.add_memory()
  → AssistantResponse
```

**Memory persistence:**
- NOT automatic for every conversation turn.
- Heuristic-based: keywords like "remember", "note", "important" trigger persistence.
- Can be forced with `persist_memory=True`.
- Failures are silently logged (never crash the assistant).

**Error handling:**
- Empty/whitespace messages → `ValueError`
- Missing sessions → `SessionNotFoundError`
- Provider failures → `ProviderError`
- Memory failures → silently ignored

## Architecture Changes

```
User message
  → ChatAssistant
    → ContextBuilder (memory + history + mode → messages)
    → AIProvider.chat(messages)
    → Session (store user + assistant messages)
    → MemoryStore (optional persistence)
  → AssistantResponse
```

This establishes the canonical architecture:
```
API → Assistant → ChatAssistant/BusinessAssistant/CodingAgent → ContextBuilder → AIProvider → Qwen
```

## Files Changed

| File | Status | Purpose |
|------|--------|---------|
| `src/assistant/context.py` | **NEW** | ContextBuilder, Context dataclass, MODE_INSTRUCTIONS |
| `src/assistant/chat.py` | **NEW** | ChatAssistant, AssistantResponse |
| `src/assistant/__init__.py` | UPDATED | Exports ContextBuilder, Context, ChatAssistant, AssistantResponse |
| `tests/test_assistant_context.py` | **NEW** | 36 tests for ContextBuilder |
| `tests/test_assistant_chat.py` | **NEW** | 26 tests for ChatAssistant |
| `docs/ARCHITECTURE.md` | UPDATED | Added assistant layer, context builder, session, error hierarchy, data flow |
| `docs/DEVELOPMENT_STATUS.md` | UPDATED | Updated test counts, component status |

## Tests

| Metric | Value |
|--------|-------|
| Previous baseline | 239/239 |
| New tests (context) | 36 |
| New tests (chat) | 26 |
| Total new | 62 |
| **Total passing** | **301/301** |
| Failed | 0 |
| Skipped | 0 |

## Known Limitations

1. **Memory keyword heuristic**: Memory persistence uses a simple keyword heuristic ("remember", "note", etc.). More sophisticated extraction (e.g., semantic importance scoring) is deferred to later phases.
2. **Session persistence**: Sessions are in-memory only. Process restart loses all session state. Disk persistence deferred to later phases.
3. **Token counting**: ContextBuilder uses character counts as a proxy for token limits. True token counting requires tokenizer integration.
4. **Async threading**: ChatAssistant uses `asyncio.run()` with thread pool fallback for synchronous callers. Native async API endpoints will simplify this.
5. **No streaming**: ChatAssistant does not support streaming responses yet. Streaming requires provider + assistant layer changes.

## Security Considerations

- ContextBuilder does not expose raw memory entries in API responses.
- Session IDs are UUID-based and unpredictable.
- Memory store operations are sandboxed — failures never propagate to the user.
- No secrets are included in context messages.

## Next Recommended Step

**Step 5-6: Business Assistant**
- BusinessAssistant with structured analysis modes
- Business context injection
- Business-specific memory and tools
