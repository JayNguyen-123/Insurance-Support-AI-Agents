# Insurance Support AI Agents

A production-grade, multi-agent AI system for insurance customer support: a **Supervisor** agent routes each request to a **Policy**, **Billing**, **Claims**, **General Help** (RAG/FAQ), or **Human Escalation** specialist, built on [LangGraph](https://github.com/langchain-ai/langgraph) and served as a **FastAPI** REST API.

This is a rewrite of an exploratory Jupyter/Colab notebook (`Insurance_Support_AI_Agents.ipynb`) into a deployable service: modular code, environment-driven configuration, structured logging, retries, a real test suite, and a Docker image -- plus fixes for several bugs the notebook's linear "run all cells" execution style let slide. See **[What changed from the notebook](#what-changed-from-the-notebook)** below.

## Architecture

```
Client
  │  POST /api/v1/chat {session_id?, message}
  ▼
FastAPI (app/main.py)
  │
  ▼
Conversation Service (app/services/conversation_service.py)
  │  loads/creates a Session (app/services/session_store.py)
  │  resumes a paused clarification, or starts a fresh turn
  ▼
LangGraph app (app/agents/graph.py)
  │
  ├─► Supervisor Agent ──► Policy Agent ────┐
  │        │              Billing Agent ────┤  (each reports back to
  │        │              Claims Agent ─────┤   the Supervisor)
  │        │              General Help ─────┘
  │        │
  │        ├─► needs clarification? → pause, return question to caller
  │        ├─► too many iterations? → Human Escalation Agent
  │        └─► done? → Final Answer Agent → clean response
  │
  ▼
SQLite (policies/billing/claims/customers)   ChromaDB (FAQ vectors)
```

Each specialist agent calls OpenAI (`gpt-5-mini` by default) with function-calling tools backed by `app/db/repository.py` (parameterized SQL against the synthetic SQLite database) or, for General Help, a FAQ vector search (`app/vectorstore/faq_store.py`).

## What changed from the notebook

The original notebook was a working prototype; turning it into a service required fixing several bugs that only surface once you stop clicking "Run All" in Colab:

- **Blocking `input()` for clarification questions.** The notebook's `ask_user()` called Python's `input()`, which blocks a thread waiting on a keyboard that will never write to it -- fatal in a web server. The supervisor now returns `needs_clarification=True` and pauses the graph; the API returns the question to the caller and resumes the conversation on the next request with the answer (`app/agents/supervisor.py`, `app/services/conversation_service.py`).
- **A syntax error in `decide_next_agent`** (`if state.get("needs_clarification": return ...)`) would have raised `SyntaxError` the moment that cell ran outside of the specific state the notebook happened to be in.
- **A typo (`policy_agennt`) silently dropped `policy_agent`'s edge back to the supervisor**, leaving it a dead-end node.
- **`policy_agent_node` and `claims_agent_node` never appended their answers to `conversation_history`**, so the supervisor's next routing decision couldn't see what they'd said (only `billing_agent_node` and `general_help_agent_node` did this correctly). All four specialists now update history consistently.
- **The supervisor's iteration counter double/triple-counted a single clarification round-trip** (ask → merge → route was three separate invocations, each incrementing the counter), which could burn the entire iteration budget on one clarifying question and force an unwanted escalation on the very next real decision. Resuming with an answer and making the routing decision now happen in one call.
- **`'OH' 'GA'`** (missing comma) in the sample-data generator was silently concatenated by Python into a bogus state code `'OHGA'`.
- **The DROP script referenced a table named `billings`** while the table is created as `billing`, so re-seeding an already-seeded database silently failed to clear old rows.
- **`trace_agent`'s `span.set_attribute(Status(...))`** should have been `span.set_status(...)`; span status was never actually being recorded.
- **`getpass.getpass(...)` at import time** for the Phoenix tracing endpoint blocks forever in any non-interactive process. Tracing is now fully optional and config-driven (`PHOENIX_COLLECTOR_ENDPOINT`); the app runs normally with it unset.
- **Loading the FAQ dataset from Hugging Face at import time** meant the app couldn't start without network access to the Hub. It now seeds from a small bundled CSV by default and only attempts the Hugging Face download when explicitly enabled, falling back automatically on any failure.

None of this changes the product behavior described in the notebook -- it makes that behavior actually work outside of a notebook.

## Quickstart

### Option A: Docker (recommended)

```bash
cp .env.example .env
# edit .env and set OPENAI_API_KEY

docker compose up --build
```

The API is now at `http://localhost:8000`. The database and FAQ vector store are seeded automatically on first boot (see `app/main.py`'s startup handler) and persisted in a named Docker volume.

### Option B: Local Python

Requires Python 3.11+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt   # or requirements.txt for a runtime-only install

cp .env.example .env
# edit .env and set OPENAI_API_KEY

python scripts/seed_db.py             # one-time: build the SQLite DB + FAQ vector store
uvicorn app.main:app --reload
```

### Try it

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the premium on my policy POL000001?"}'
```

If the supervisor needs more information, the response comes back with `"requires_clarification": true` and the question in `"reply"`. Send another request with the **same `session_id`** and the answer as `"message"` to continue.

## API

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness/readiness: DB connectivity, vector store connectivity, whether an OpenAI key is configured. |
| `/api/v1/chat` | POST | `{session_id?: string, message: string}` → `{session_id, reply, requires_clarification, requires_human_escalation, done, agent_used}`. Omit `session_id` to start a new conversation. |
| `/api/v1/sessions/{session_id}/reset` | POST | Clears a session's history and any paused clarification state. |

Interactive OpenAPI docs are available at `/docs` once the server is running.

## Configuration

All configuration is environment-driven (`app/config.py`, `pydantic-settings`) -- see `.env.example` for the full list and defaults. Nothing is hardcoded the way the notebook hardcoded its Colab secret lookup, DB filename, and tracing endpoint prompt.

Notable settings:

- `OPENAI_MODEL` (default `gpt-5-mini`) and `OPENAI_MAX_RETRIES` / `OPENAI_REQUEST_TIMEOUT_SECONDS` for LLM call resilience.
- `SUPERVISOR_MAX_ITERATIONS` (default `5`) bounds how many supervisor decision points (ask / route / confirm-done) one logical exchange can take before forcing a human escalation -- see the comment in `app/config.py` for why this needs headroom above 3.
- `FAQ_USE_HUGGINGFACE_DATASET` (default `false`) -- set to `true` to seed the FAQ vector store from the real `deccan-ai/insuranceQA-v2` dataset on Hugging Face instead of the bundled sample set (`data/faq_seed.csv`).
- `PHOENIX_COLLECTOR_ENDPOINT` -- set to enable Arize Phoenix / OpenTelemetry agent tracing; leave unset to disable it entirely.

## Testing

```bash
pip install -r requirements-dev.txt
pytest
```

The suite covers: DB repository functions against a seeded temp SQLite database, the FAQ vector store, the supervisor's routing/JSON-parsing/escalation logic (including a dedicated regression test proving a clarification question no longer blocks or double-counts iterations), the LangGraph wiring (including regression tests for the two graph-construction bugs above), the full conversation service (session creation, pause/resume across two calls, human escalation), and the FastAPI endpoints end-to-end -- all against a scripted fake OpenAI client, so no real API key or network access is required to run the tests.

## Known limitations & natural next steps

This is a **solid engineering baseline**, not a fully hardened enterprise deployment. Deliberately out of scope, and worth knowing about before you go further:

- **SQLite + in-memory sessions are single-process.** They're fine for a demo or a single-instance deployment, but won't work correctly behind multiple worker processes/instances without changes. Swap `app/db/session.py` for a PostgreSQL connection pool and `app/services/session_store.py` for Redis (or another shared store) to scale out -- both were written with a narrow, documented interface specifically so that swap is contained.
- **No authentication, rate limiting, or per-tenant isolation** on the API. Add these (e.g. an API gateway, API keys, or OAuth) before exposing this beyond a trusted network.
- **No PII redaction or prompt-injection guardrails** on user input or LLM output beyond what's in the prompts themselves.
- **No native LangGraph checkpointer.** The clarification pause/resume is implemented by hand (storing/reloading plain `GraphState` dicts in the session store) rather than via LangGraph's built-in `interrupt()`/checkpointer machinery, which is the more scalable approach for complex multi-turn human-in-the-loop flows at scale, but adds real complexity for a first production cut.
- **The synthetic sample data** (`app/db/seed_data.py`) is randomly generated for demo purposes and does not represent real customers, policies, or claims.

## Project layout

```
app/
  agents/        Prompts, LangGraph node functions, tool schemas, and graph wiring
  db/            SQLite schema, seed data, connection management, repository (tool) functions
  llm/           OpenAI client wrapper with retries
  models/        Pydantic request/response models
  services/      Session store + conversation orchestration
  vectorstore/   ChromaDB-backed FAQ store
  config.py      Environment-driven settings
  logging_config.py
  tracing.py     Optional Phoenix/OpenTelemetry tracing
  main.py        FastAPI app
data/
  faq_seed.csv   Bundled fallback FAQ dataset
scripts/
  seed_db.py     CLI to (re)build the DB and FAQ vector store
tests/           pytest suite (scripted fake OpenAI client, no network required)
docker/
  Dockerfile
docker-compose.yml
```
