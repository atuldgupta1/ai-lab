# ai-lab

A reusable LLM harness for learning AI engineering. It wraps the Anthropic SDK
with typed configuration, per-call cost logging, automatic retries with
exponential backoff, and an async path for concurrent calls.

This is the Phase 0 foundation — the base layer that later projects (RAG,
agents, orchestration) build on top of.

---

## Requirements

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/) for environment and dependency management
- An Anthropic API key

---

## Setup

1. **Install dependencies** (reads `pyproject.toml` / `uv.lock`):

   ```bash
   uv sync
   ```

2. **Add your Anthropic API key** to a `.env` file in the project root:

   ```
   ANTHROPIC_API_KEY=sk-ant-...
   ```

   > `.env` is git-ignored and must never be committed.

---

## Usage

### Synchronous

```python
from src.llm.client import complete

print(complete("Explain recursion in one sentence."))
```

### Asynchronous (many calls at once)

```python
import asyncio
from src.llm.client import acomplete

async def main():
    results = await asyncio.gather(
        acomplete("Name a fruit. One word."),
        acomplete("Name a color. One word."),
        acomplete("Name an animal. One word."),
    )
    print(results)

asyncio.run(main())
```

### Custom configuration

```python
from src.llm.client import complete, LLMConfig

config = LLMConfig(
    model="claude-sonnet-4-6",
    temperature=0.3,
    max_tokens=512,
)
print(complete("Summarise the plot of Hamlet.", config=config))
```

---

## Features

- **Typed config** — `LLMConfig` (Pydantic) validates settings on creation;
  e.g. an out-of-range temperature is rejected immediately.
- **Clear failures** — a missing API key raises a readable error up front
  instead of a cryptic SDK error later.
- **Cost logging** — every call logs input/output tokens and estimated USD cost,
  so token usage is always visible.
- **Retries with backoff** — transient errors (rate limits, connection issues,
  5xx) are retried automatically with exponential backoff via `tenacity`.
  Non-transient errors are not retried.
- **Async path** — `acomplete()` plus `asyncio.gather` runs many calls
  concurrently; throughput is capped by the slowest call, not the sum.

---

## Models

Defaults to **Claude Haiku 4.5** (the cheapest model) for low-cost development.
Override per call via `LLMConfig(model=...)`.

| Model           | Input ($/M) | Output ($/M) | Use for                     |
|-----------------|-------------|--------------|-----------------------------|
| Haiku 4.5       | 1           | 5            | Development / testing       |
| Sonnet 4.6      | 3           | 15           | Best price/quality balance  |
| Opus 4.8        | 5           | 25           | Hardest tasks only          |

> Rates are per million tokens and can change — check the Anthropic console
> for current pricing.

---

## Testing

```bash
uv run pytest -v
```

Most tests mock the API, so they run instantly and cost nothing. They cover the
cost math, config validation, the missing-key error path, and the text-extraction
logic in `complete()`.

---

## Project structure

```
ai-lab/
├── .env                  # API key (git-ignored, never committed)
├── .gitignore
├── pyproject.toml        # dependencies + pytest config
├── README.md
├── src/
│   └── llm/
│       ├── __init__.py
│       └── client.py     # the harness
└── tests/
    └── test_client.py
```