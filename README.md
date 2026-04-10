# 🏠 HouseGPT

> **A production-grade, multimodal AI assistant** built on LangGraph, powered by Dr. House's personality — combining long-term memory, RAG-augmented medical knowledge, speech/image understanding, and persistent multi-session conversation.

---

## 📋 Table of Contents

- [Architecture Overview](#architecture-overview)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Environment Variables](#environment-variables)
- [Running the Application](#running-the-application)
- [API Reference](#api-reference)
- [LangGraph Workflow](#langgraph-workflow)
- [Memory System](#memory-system)
- [Known Issues & Fixes](#known-issues--fixes)
- [Performance Notes](#performance-notes)

---

## Architecture Overview

```
                         ┌──────────────────────────────────────┐
                         │            FastAPI + Gunicorn         │
                         │         (UvicornWorker, 4 workers)    │
                         └────────────────┬─────────────────────┘
                                          │
                    ┌─────────────────────┼──────────────────────┐
                    │                     │                       │
             POST /chat            POST /chat/audio       POST /chat/image
                    │                     │                       │
                    └─────────────────────┼──────────────────────┘
                                          │
                              ┌───────────▼──────────┐
                              │     LangGraph Agent   │
                              │  (Compiled StateGraph) │
                              └───────────┬───────────┘
                                          │
           ┌─────────────────────────────┬┴──────────────────────────┐
           │                             │                            │
   ┌───────▼──────┐           ┌──────────▼────────┐       ┌─────────▼───────┐
   │ Memory Nodes │           │   Router Node      │       │  Context Node   │
   │ (inject/     │           │ (small LLM →       │       │ (schedule-based │
   │  extract)    │           │  conversation/rag) │       │  activity)      │
   └───────┬──────┘           └──────────┬─────────┘       └─────────┬───────┘
           │                             │                            │
           └─────────────────────────────┴────────────────────────────┘
                                          │
                              ┌───────────▼────────────┐
                              │      Dispatch Node      │
                              └──────────┬──┬───────────┘
                                         │  │
                          ┌──────────────┘  └─────────────┐
                          │                               │
               ┌──────────▼──────────┐      ┌────────────▼────────────┐
               │  Conversation Node  │      │    Medical RAG Node      │
               │  (character LLM)    │      │  (hybrid dense+sparse)   │
               └──────────┬──────────┘      └────────────┬────────────┘
                          │                               │
                          └───────────────┬───────────────┘
                                          │
                          ┌───────────────▼───────────────┐
                          │    Summarize (if needed)       │
                          │    → PostgreSQL Checkpoint     │
                          └───────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| API Framework | FastAPI + Gunicorn + UvicornWorker |
| Agent Orchestration | LangGraph (StateGraph) |
| LLM Provider | Ollama (configurable small/large models) |
| Short-Term Memory | PostgreSQL via `AsyncPostgresSaver` (LangGraph checkpoint) |
| Long-Term Memory | Qdrant (vector search, cosine similarity) |
| Medical RAG | Qdrant hybrid search (dense + SPLADE sparse, RRF fusion) |
| Embeddings | Ollama (local, configurable model) |
| Speech-to-Text | OpenAI Whisper (local) |
| Image-to-Text | Ollama vision model |
| Rate Limiting | SlowAPI |
| Logging | Custom `AppLogger` |

---

## Project Structure

```
house_gpt/
├── api/
│   └── v1/
│       ├── app.py                  # FastAPI app, startup/shutdown lifecycle
│       └── routers/
│           └── chat.py             # /chat, /chat/audio, /chat/image endpoints
├── agent/
│   ├── chains.py                   # LRU-cached LangChain chains
│   ├── graph/
│   │   ├── graph.py                # create_workflow_graph() — LangGraph builder
│   │   ├── nodes/                  # One file per node
│   │   │   ├── conversation_node.py
│   │   │   ├── medical_rag_node.py
│   │   │   ├── memory_nodes.py
│   │   │   ├── router_node.py
│   │   │   ├── context_injection_node.py
│   │   │   └── summarize_conversation_node.py
│   │   └── edges.py                # Routing logic: select_workflow, should_summarize
│   ├── helpers/
│   │   ├── model_factory.py        # get_small_model(), get_large_model()
│   │   └── formatter.py            # AsteriskRemovalParser, build_rag_context
│   └── prompts.py                  # All system prompts
├── memory/
│   ├── ltm/
│   │   ├── vector_store.py         # VectorStore (Qdrant + Ollama embeddings)
│   │   └── memory_manager.py       # MemoryManager (extract + inject)
│   ├── rag/
│   │   └── rag_memory.py           # MedicalBooksRAG (hybrid search)
│   └── models.py                   # Memory, RAG dataclasses
├── multimodal/
│   ├── speech
        └── speech_to_text.py       # SpeechToText (Whisper)
│   └── image
        └── image_to_text.py        # ImageToText (vision LLM)
├── schedules/
│   └── context_generation.py       # ScheduleContextGenerator
├── services/
│   ├── graph_services.py           # invoke_graph()
│   └── image_services.py           # get_image_type()
├── states/
│   ├── house.py                    # AIHouseState (TypedDict)
│   ├── memory.py                   # MemoryAnalysis
│   └── response.py                 # RouterResponse
└── core/
    ├── settings.py                 # Pydantic settings
    ├── logger.py                   # AppLogger
    ├── exceptions.py               # SpeechToTextError, etc.
    └── graph_instance.py           # Global graph/pool singletons
```

---

## Features

- **Multi-turn conversation** with PostgreSQL-backed LangGraph checkpoints (persistent across restarts)
- **Smart routing**: lightweight LLM decides `conversation` vs `medical_rag` per message
- **Long-term memory**: automatically extracts and recalls personal facts about the user via Qdrant
- **Medical RAG**: hybrid dense + SPLADE sparse search with RRF fusion over medical books
- **Speech input**: Whisper transcription (async, temp-file based)
- **Image input**: vision LLM describes images before routing through the main graph
- **Conversation summarization**: auto-compresses history when message count exceeds threshold
- **Schedule-aware context**: injects Dr. House's current activity based on time of day
- **Rate limiting**: per-IP, configurable per endpoint
- **Health endpoints**: `/health` (simple) and `/health/detailed` (checks Postgres + Qdrant + graph)

---

## Prerequisites

- Python 3.11+
- Poetry
- PostgreSQL (running and accessible)
- Qdrant (running, local or cloud)
- Ollama (running locally)
- CUDA GPU (**minimum 6 GB VRAM** for Whisper + embeddings — see [Known Issues](#known-issues--fixes))

---

## Installation

```bash
# Clone the repo
git clone https://github.com/yourorg/house-gpt.git
cd house-gpt

# Install dependencies
poetry install

# Pull the embedding model in Ollama
ollama pull qwen3-embedding:8b   # or whichever EMBEDDING_MODEL_NAME you set

# Create the Qdrant collections (run once)
poetry run python scripts/setup_qdrant.py

# Create log directory
mkdir -p logs
```

---

## Environment Variables

Create a `.env` file at the project root:

```env
# ── LLM ───────────────────────────────────────────────
SMALL_TEXT_MODEL_NAME=qwen2.5:7b
LARGE_TEXT_MODEL_NAME=qwen3:8b
ITT_MODEL_NAME=qwen3-vl:8b

# ── Speech ────────────────────────────────────────────
STT_MODEL_NAME=base          # whisper model size: tiny | base | small | medium

# ── Embeddings ────────────────────────────────────────
EMBEDDING_MODEL_NAME=qwen3-embedding:8b

# ── PostgreSQL ────────────────────────────────────────
POSTGRES_URI=postgresql://user:password@localhost:5432/housegpt
POSTGRES_MIN_CONNECTIONS=2
POSTGRES_MAX_CONNECTIONS=10

# ── Qdrant ────────────────────────────────────────────
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=                # leave empty for local unauthenticated

# ── Memory ────────────────────────────────────────────
MEMORY_TOP_K=5
SHORT_TERM_MEMORY_DB_PATH=./data/memory.db   # legacy, unused if Postgres is set

# ── Agent ─────────────────────────────────────────────
ROUTER_MESSAGES_TO_ANALYZE=5
TOTAL_MESSAGES_SUMMARY_TRIGGER=20
TOTAL_MESSAGES_AFTER_SUMMARY=6

# ── PyTorch (avoids GPU memory fragmentation) ─────────
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

---

## Running the Application

### Development

```bash
poetry run uvicorn house_gpt.api.v1.app:app --reload --port 8000
```

### Production (single GPU — recommended)

> ⚠️ **Important**: Use `-w 1` on machines with limited VRAM. Running 4 workers loads Whisper + embeddings **4 times** — easily exceeding 6 GB VRAM. See [Known Issues](#known-issues--fixes).

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
poetry run gunicorn house_gpt.api.v1.app:app \
  -w 1 \
  -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 120 \
  --keep-alive 5 \
  --access-logfile logs/access.log \
  --error-logfile logs/error.log
```

### Production (multi-worker, CPU-only Whisper)

If you want multiple workers, offload Whisper to CPU:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
WHISPER_DEVICE=cpu \
poetry run gunicorn house_gpt.api.v1.app:app \
  -w 4 \
  -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 120 \
  --keep-alive 5 \
  --access-logfile logs/access.log \
  --error-logfile logs/error.log
```

### Docker (optional)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
RUN pip install poetry
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-dev
COPY . .
RUN mkdir -p logs
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CMD ["poetry", "run", "gunicorn", "house_gpt.api.v1.app:app", \
     "-w", "1", "-k", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", "--timeout", "120"]
```

---

## API Reference

Base URL: `http://localhost:8000/api/v1`

### `POST /chat`

Send a text message.

**Request**
```json
{
  "message": "What are the symptoms of lupus?",
  "session_id": "user-abc-123"
}
```

**Response**
```json
{
  "reply": "It's never lupus. Except when it is...",
  "session_id": "user-abc-123",
  "route_used": "medical_rag",
  "transcription": null
}
```

**Rate limit:** 20 requests/minute per IP

---

### `POST /chat/audio`

Send an audio file for transcription + response.

**Form data**
| Field | Type | Description |
|---|---|---|
| `session_id` | string | Session identifier |
| `audio` | file | Audio file (wav, mp3, ogg, webm, mp4 — max 10 MB) |

**Response**
```json
{
  "reply": "...",
  "session_id": "user-abc-123",
  "route_used": "conversation",
  "transcription": "What are the symptoms of lupus?"
}
```

**Rate limit:** 10 requests/minute per IP

---

### `POST /chat/image`

Send an image for analysis + response.

**Form data**
| Field | Type | Description |
|---|---|---|
| `session_id` | string | Session identifier |
| `image` | file | Image file (jpeg, png, webp, gif — max 5 MB) |
| `prompt` | string | Optional context prompt for image analysis |

**Response**
```json
{
  "reply": "...",
  "session_id": "user-abc-123",
  "route_used": "conversation",
  "image_description": "The image shows an MRI scan of..."
}
```

**Rate limit:** 10 requests/minute per IP

---

### `GET /health`

```json
{ "status": "ok" }
```

### `GET /health/detailed`

```json
{
  "status": "ok",
  "checks": {
    "postgres": "ok",
    "qdrant": "ok",
    "graph": "ok"
  }
}
```

Returns `503` with `"status": "degraded"` if any dependency is down.

---

## LangGraph Workflow

The agent runs a `StateGraph` over `AIHouseState`. All four preparation nodes run in **parallel** from `START`:

```
START
  ├── memory_extraction_node   (fire-and-forget background task)
  ├── router_node              (classifies: conversation | rag)
  ├── context_injection_node   (injects schedule/activity context)
  └── memory_injection_node    (fetches relevant long-term memories)
          ↓ (all join at)
      dispatch_node
          ↓ (conditional)
  ┌── conversation_node        (character LLM response)
  └── medical_rag_node         (hybrid RAG + character LLM response)
          ↓ (conditional)
  ┌── summarize_conversation_node   (if > TOTAL_MESSAGES_SUMMARY_TRIGGER)
  └── END
```

**Key design decisions:**

- `memory_extraction_node` fires a background `asyncio.Task` — it never blocks the response path
- `get_router_chain()` and `get_character_response_chain()` are `@lru_cache`'d — chains are built once
- `create_workflow_graph()` is `@lru_cache(maxsize=1)` — compiled graph is a singleton per process
- `AsyncPostgresSaver` provides thread-safe, multi-session state persistence

---

## Memory System

### Short-Term Memory (Per Session)

Managed by LangGraph's `AsyncPostgresSaver`. Each `thread_id` (= `session_id`) stores the full message history and state. Auto-summarization kicks in at `TOTAL_MESSAGES_SUMMARY_TRIGGER` messages, keeping the last `TOTAL_MESSAGES_AFTER_SUMMARY` messages in the active window.

### Long-Term Memory (Per User)

Managed by `MemoryManager` + `VectorStore`:

1. After each human message, `memory_extraction_node` calls a structured LLM to extract personal facts
2. Before each response, `memory_injection_node` retrieves the top-K most relevant memories from Qdrant
3. Near-duplicate memories are detected (cosine similarity ≥ 0.7) and upserted rather than duplicated
4. Memories are scoped per `user_id` via Qdrant payload filters

### Medical RAG

`MedicalBooksRAG` uses **hybrid retrieval**:
- **Dense**: Ollama embeddings → cosine search
- **Sparse**: SPLADE (`prithivida/Splade_PP_en_v1`) → sparse inverted index
- **Fusion**: Reciprocal Rank Fusion (RRF) re-ranks results from both retrievers
- Results below `SIMILARITY_THRESHOLD = 0.75` are filtered out

---

## Known Issues & Fixes

### ❌ CUDA Out of Memory with `-w 4`

**Symptom**
```
CUDA out of memory. Tried to allocate 20.00 MiB. GPU 0 has a total capacity of 5.67 GiB...
gunicorn.errors.HaltServer: <HaltServer 'Worker failed to boot.' 3>
```

**Root cause**

Gunicorn spawns 4 separate processes. Each process loads Whisper + Ollama embeddings + SPLADE into GPU memory independently. On a 5.67 GB GPU, 4 × ~1.4 GB = ~5.6 GB is already the hard limit before any inference happens.

**Fix (recommended): Use a single worker**

```bash
poetry run gunicorn house_gpt.api.v1.app:app \
  -w 1 \
  -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 120
```

One `UvicornWorker` with asyncio handles high concurrency without additional GPU copies. This is the correct approach for GPU-bound services.

**Fix (optional): Reduce memory fragmentation**

Add to your environment before starting gunicorn:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

**Fix (optional): Run Whisper on CPU**

If you must use multiple workers, move Whisper to CPU in `speech_to_text.py`:

```python
self.model = whisper.load_model(settings.STT_MODEL_NAME, device="cpu")
```

This frees ~500 MB–1 GB of VRAM per worker at the cost of slower transcription.

---

## Performance Notes

- All LangGraph nodes that run at `START` execute in parallel — total latency is `max(node_latency)` not `sum`
- LLM chains are cached with `@lru_cache` — no rebuild cost on repeated requests
- Qdrant queries run in a thread executor (`run_in_executor`) to avoid blocking the event loop
- Memory injection has a 10-second timeout with graceful fallback (empty memory context)
- Graph invocation has a 60-second hard timeout at both the service layer and endpoint layer
- `AsyncConnectionPool` is shared across the entire process lifetime, opened once at startup

---

## 🚀 Roadmap

The following capabilities are planned for upcoming releases:

**Audio Generation** — Dr. House will be able to respond with synthesized speech, giving the assistant a full voice interface. The plan is to integrate a TTS model (likely a local one via Coqui or a hosted API) directly into the response pipeline, with the audio returned alongside the text reply.

**Image Generation** — The assistant will gain the ability to generate images on demand — whether to illustrate a medical concept, visualize a diagnosis, or respond creatively. This will be routed through the existing workflow graph as a new node, triggered by the router when the user's intent calls for visual output.

Both features will extend the existing multimodal architecture without breaking current endpoints.