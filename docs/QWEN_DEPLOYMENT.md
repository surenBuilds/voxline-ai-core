# Qwen Deployment

Phase 6 — Qwen2.5-0.5B-Instruct as the default deployment provider.

## Overview

Qwen2.5-0.5B-Instruct is the default AI provider for Voxline AI Core.
It is stored locally at `models/Qwen2.5-0.5B-Instruct/` and runs on CPU.

## Configuration

### Default Provider

The default provider is now `qwen` (changed from `local_hf`):

```python
# src/config/settings.py
DEFAULTS = {
    "AI_PROVIDER": "qwen",
    "AI_MODEL_PATH": "models/Qwen2.5-0.5B-Instruct",
    ...
}
```

### Environment Variable Override

```bash
# Use native Voxline model
AI_PROVIDER=native python serve_v04.py

# Use Qwen (default)
python serve_v04.py
# or
python serve_v04.py --provider qwen
```

## Starting the Server

```bash
# Default (Qwen)
python serve_v04.py

# Explicit Qwen
python serve_v04.py --provider qwen

# Native Voxline (deprecated, for reference only)
python serve_v04.py --provider native

# Custom host/port
python serve_v04.py --host 0.0.0.0 --port 8000
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check with provider status |
| `/models` | GET | Model info (provider, params, vocab, context) |
| `/chat` | POST | Multi-turn chat (OpenAI-style messages) |
| `/generate` | POST | Single prompt generation |

### Chat Request

```json
{
  "message": "What is the capital of France?",
  "max_tokens": 100,
  "temperature": 0.8
}
```

### Generate Request

```json
{
  "prompt": "The capital of France is",
  "max_tokens": 50,
  "temperature": 0.7
}
```

## Memory Usage

- Model: 494M parameters, ~942MB on disk
- Runtime: ~1.5GB RAM (float32 on CPU)
- Safe to run on systems with 4GB+ RAM

## Performance Baseline

Measured on CPU (Intel/AMD, no GPU):

| Metric | Value |
|--------|-------|
| Load time | ~15s |
| First inference | ~8s |
| Average latency | ~3-5s per response |
| Throughput | ~15-25 tokens/sec |

## Chat Quality

Qwen2.5-0.5B-Instruct provides:
- Bilingual support (English + Armenian)
- Instruction following
- Context retention in multi-turn conversations
- Reasoning capabilities for simple tasks

## Resource Safety

- Model is loaded lazily (on first request, not on server start for evaluation)
- Singleton pattern prevents duplicate loading
- Memory limit: provider raises on load failure if RAM insufficient

## Provider Selection

The `ProviderFactory` class handles provider creation:

```python
from src.config.settings import get_config
from src.providers.factory import ProviderFactory

config = get_config()  # AI_PROVIDER=qwen
provider = ProviderFactory.create(config)
```

## Limitations

- CPU-only inference (no GPU acceleration)
- 0.5B parameters limits complex reasoning
- No fine-tuning or retraining (Phase 6 constraint)
- Armenian language support is limited
