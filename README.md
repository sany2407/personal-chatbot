# Portfolio Agentic RAG Assistant — Backend

FastAPI + LangGraph + ChromaDB backend for answering visitor questions using approved portfolio knowledge.

## Requirements
- Python 3.11 to 3.14 (3.15 has no `onnxruntime` wheels yet, and `chromadb` requires it)
- A Gemini API key (used for both chat and embeddings), or a Gemini key plus an API key for an OpenAI-compatible chat provider (Groq, OpenAI, OpenRouter, ...)
- No local model runtime: no Ollama, no Sentence Transformers, no torch

## Local setup

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell:
# .venv\\Scripts\\Activate.ps1

pip install -r requirements.txt
cp .env.example .env
```

When you change the Python version, delete the virtualenv first (`rm -rf .venv`).
Recreating it over an existing one keeps the old packages, and native wheels from the
previous interpreter fail at import time with errors such as
`ImportError: cannot import name 'ArgsKwargs' from 'pydantic_core._pydantic_core'`.

Edit `.env` and configure the provider you want to use (see below).
Keep API keys on the backend only.

### CORS

Three settings build the allow-list:

```env
CORS_ORIGINS_RAW=http://localhost:3000,http://127.0.0.1:3000
CORS_ORIGIN_REGEX=                 # extra patterns, e.g. ^https://sany\.dev$
CORS_ALLOW_PREVIEWS=true           # *.vercel.app + any localhost port
```

- `CORS_ORIGINS_RAW` is an exact allow-list.
- `CORS_ALLOW_PREVIEWS` (default on) always allows `https://*.vercel.app` and
  any localhost/127.0.0.1 port, so production and preview deployments work
  without per-deploy configuration.
- `CORS_ORIGIN_REGEX` **adds** patterns; it never removes the defaults. A stale
  value in a hosting dashboard therefore cannot lock the frontend out. Set
  `CORS_ALLOW_PREVIEWS=false` to serve only the exact origins and patterns.

`/health` reports the effective policy, including the combined regex, so a
blocked browser origin is diagnosable without reading logs.

## Add portfolio knowledge

Replace the sample files under `knowledge/` with accurate, public, approved content.
Supported source formats are Markdown (`.md`) and plain text (`.txt`).

Index the knowledge:

```bash
python -m app.rag.ingest
```

Run the API:

```bash
uvicorn app.main:app --reload
```

Open:
- Health: http://localhost:8000/health
- Interactive API docs: http://localhost:8000/docs

## Test the chat endpoint

```bash
curl -X POST http://localhost:8000/api/v1/chat \\
  -H "Content-Type: application/json" \\
  -d '{"question":"What technologies are listed in the portfolio?"}'
```

## Conversation memory

The agent keeps a per-session conversation buffer using LangGraph's SQLite
checkpointer. Every `session_id` maps to one thread, and the turn history
(question + answer) is stored in a SQLite file:

```env
CHECKPOINT_DB_PATH=./.memory/checkpoints.sqlite
MAX_HISTORY_TOKENS=2000
```

- The history survives restarts (it is on disk, not in process RAM). The
  `.memory/` directory is mounted as a volume in `docker-compose.yml`, so
  conversations also survive container recreation.
- `MAX_HISTORY_TOKENS` caps how much of the recent conversation is replayed
  into the prompt; older turns are dropped, so the buffer cannot grow without
  bound.
- The API stays stateless: pass the same `session_id` back to continue a
  conversation, or omit it to start a new one.
- History is used only for intent (pronouns, follow-ups). The system prompt
  forbids treating earlier assistant answers as evidence — every factual claim
  must still come from the freshly retrieved evidence.

SQLite is a single-instance store: run one worker, or move to a Postgres
checkpointer (`langgraph-checkpoint-postgres`) before scaling out.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `POST` | `/api/v1/chat` | Native chat endpoint. Body: `{"question": "...", "session_id": "..."}` |
| `GET` | `/v1/models` | OpenAI-compatible model list |
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat completion |

`/api/v1/chat` also accepts an OpenAI-style `messages` array, so both bodies work:

```json
{"messages": [{"role": "user", "content": "What technologies are listed?"}]}
```

The `/v1/*` routes let standard chat clients (Open WebUI, LobeChat, custom widgets)
point at this backend with `base_url = http://localhost:8000/v1` and the model name
shown by `GET /v1/models`. Streaming (`"stream": true`) is not implemented yet and
returns a normal JSON completion.

The `my_portfolio` frontend ships a floating `SANY_AI` widget that uses
`POST /api/v1/chat` and keeps a per-visit `session_id`.

## Gemini (Google AI API)

Get a key from https://aistudio.google.com/apikey and configure:

```env
LLM_PROVIDER=gemini
LLM_MODEL=gemini-3.6-flash
GEMINI_API_KEY=your-gemini-key
```

`GOOGLE_API_KEY` is accepted as an alias, and `LLM_API_KEY` is used as a fallback
when `LLM_PROVIDER=gemini`.

Model notes:
- `gemini-3.6-flash` is the default (stable, good quality/latency balance).
  `gemini-3.5-flash-lite` is faster and cheaper for high traffic.
- Gemini 3+ models use fixed sampling defaults and ignore `LLM_TEMPERATURE`.
- `LLM_MAX_RETRIES` and `LLM_TIMEOUT` bound the provider SDK's retry/backoff
  behaviour. Without them a provider outage turns into a multi-minute hang
  instead of a fast error.

Embeddings always use the hosted Gemini embedding model. No local transformer
model is downloaded or run:

```env
EMBEDDING_PROVIDER=gemini
GEMINI_EMBEDDING_MODEL=gemini-embedding-2
```

`GEMINI_API_KEY` is used for both the chat model and the embedding model.

## OpenAI-compatible providers (Groq, OpenAI, OpenRouter, ...)

Point the base URL at the provider's API root and use one of its model IDs:

```env
LLM_PROVIDER=openai_compatible
LLM_API_KEY=your-provider-key
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL=openai/gpt-oss-120b
```

Other examples:
- Groq: `https://api.groq.com/openai/v1` (e.g. `openai/gpt-oss-120b`, `moonshotai/kimi-k2-instruct`)
- OpenAI: `https://api.openai.com/v1` (e.g. `gpt-4o-mini`)
- OpenRouter: `https://openrouter.ai/api/v1`

Gemini also exposes an OpenAI-compatible endpoint
(`https://generativelanguage.googleapis.com/v1beta/openai/`), which works with
the `openai_compatible` provider if you prefer a single code path.

If you change `GEMINI_EMBEDDING_MODEL`, delete the `.chroma` directory and re-run
`python -m app.rag.ingest`. Do not mix vectors created by different embedding
models in one collection.

## Deploy to Render

`render.yaml` describes the whole service, so the fastest path is a Blueprint:

1. Push this repository to GitHub (already at `sany2407/personal-chatbot`).
2. In the Render Dashboard: **New → Blueprint**, connect the repository, and
   apply. Render reads `render.yaml` and creates one Python web service.
3. When prompted, paste your **`GEMINI_API_KEY`** (it is marked `sync: false`,
   so it is never stored in git).
4. Wait for the first deploy, then open
   `https://<service-name>.onrender.com/health`.

The build runs `python -m app.rag.ingest`, so the Chroma index ships inside the
deploy image and cold starts do not pay for embedding 31 chunks again. If the
index is ever missing, empty, or built with a different embedding model,
`app/rag/index.py` rebuilds it at startup (`ENSURE_INDEX_ON_STARTUP=true`).

`/health` reports the index state:

```json
{
  "status": "ok",
  "index": {"status": "ready", "chunks": 31, "embedding_model": "gemini-embedding-2"},
  "index_ready": true,
  "environment": "production"
}
```

### Point the frontend at it

Set the deployed URL in the portfolio project and redeploy the frontend:

```env
NEXT_PUBLIC_CHAT_API_URL=https://<service-name>.onrender.com
```

Also add the frontend origin to `CORS_ORIGINS_RAW` (or cover it with
`CORS_ORIGIN_REGEX`) and redeploy the backend. The blueprint ships a regex that
already allows `https://*.vercel.app` and any localhost port.

### Free plan caveats

- The service sleeps after 15 minutes without traffic; the first request after
  that can take 30–60s while the instance wakes up. The frontend client waits
  up to 90s, so the widget shows a normal answer (or its predefined fallback if
  the wake-up takes longer).
- The filesystem is ephemeral. The vector index is unaffected (it is part of
  the image), but SQLite conversation history resets on every restart or
  spin-down. Visitors simply lose earlier context; nothing breaks.
- Bump to a paid instance and attach a Render disk mounted at `/app/.memory` if
  conversations must survive restarts.

### Manual equivalent

**New → Web Service → Python**, then:

```text
Build:  pip install -r requirements.txt && python -m app.rag.ingest
Start:  uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 1
Health: /health
```

Set `PYTHON_VERSION=3.12.7`, `LLM_PROVIDER=gemini`,
`LLM_MODEL=gemini-3.6-flash`, `GEMINI_EMBEDDING_MODEL=gemini-embedding-2`,
`GEMINI_API_KEY`, `CORS_ORIGINS_RAW`, and optionally `RATE_LIMIT_PER_MINUTE`.

Keep `--workers 1`: the SQLite checkpointer and the in-process rate limiter are
single-instance stores.

## Architecture

- `app/rag/ingest.py`: load, chunk and index knowledge files.
- `app/rag/index.py`: startup index check/rebuild for ephemeral filesystems.
- `app/rag/embeddings.py`: Gemini embedding model.
- `app/rag/vectorstore.py`: persistent Chroma vector store.
- `app/agent/graph.py`: LangGraph retrieval/evidence gate/answer workflow.
- `app/agent/llm.py`: configurable chat-model adapter (Gemini or OpenAI-compatible).
- `app/agent/memory.py`: SQLite checkpointer for conversation memory.
- `app/core/ratelimit.py`: per-client-IP limit for the paid chat routes.
- `app/api/routes.py`: native chat endpoint.
- `app/api/openai_routes.py`: OpenAI-compatible `/v1` endpoints.
- `knowledge/`: public portfolio facts used by the assistant.

## Important limitations before public deployment

This is a starter MVP, not a fully hardened production service:
- Conversation history is stored per `session_id` in SQLite, but there is no
  authentication, expiry, or per-visitor cap on the stored threads.
- Chat routes have a per-IP rate limit (`RATE_LIMIT_PER_MINUTE`, default 30/min)
  but there is no authentication, per-user quota, or shared (Redis) counter, so
  the limit is per instance and only a speed bump against scripted abuse.
- Re-ingestion adds stable chunk IDs but does not yet delete obsolete chunks when source files are removed or edited.
- The current evidence gate uses a simple context-length heuristic, not a semantic relevance classifier.
- `grounded` means evidence was retrieved; it is not a guarantee that every generated statement is factually correct.
- Keep proprietary project data and secrets out of `knowledge/`.
