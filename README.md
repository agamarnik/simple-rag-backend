# Simple RAG Backend

A lightweight Retrieval-Augmented Generation (RAG) API built with FastAPI, ChromaDB, and the Anthropic Claude API. Upload documents (text or PDF), then ask questions — including natural follow-ups — that get answered using only the content of those documents, with source attribution.


## How it works

This project combines several pieces:

1. **Chunking** — documents are split into sentence-aware chunks (~500 characters, with sentence-level overlap) using NLTK, so retrieval works on coherent chunks rather than arbitrary character slices.
2. **Embedding + vector search** — each chunk is converted into a vector embedding and stored in a local, persistent ChromaDB collection. Questions are embedded the same way, and Chroma's approximate nearest-neighbor search (HNSW) finds the most semantically similar chunks — not just keyword matches.
3. **Query rewriting (conversational RAG)** — for follow-up questions ("What caused it?"), a lightweight LLM call rewrites the question into a self-contained form ("What caused World War II?") *before* retrieval, using the conversation history. This fixes a common RAG failure mode where vague follow-ups retrieve irrelevant chunks, since raw vector search has no memory of its own.
4. **Multi-turn memory** — conversations are tracked by a `conversation_id`, with history stored server-side so Claude has full context across turns.
5. **Grounded generation** — retrieved chunks are inserted into a prompt with an explicit instruction to answer *only* from that context, keeping answers grounded in your documents rather than the model's general knowledge.

  ```Question → rewrite (if follow-up) → embed → vector search → top-k chunks
  → prompt with context + history → Claude → grounded, sourced answer
  ````

Documents persist to disk (`./chroma_db`) via ChromaDB's `PersistentClient`, so the vector store survives server restarts without needing to re-embed everything.

## Project structure

  simple-rag-backend/
  ├── main.py          # FastAPI app and route definitions
  ├── models.py         # Pydantic request models
  ├── auth.py           # API key verification
  ├── retrieval.py       # Query rewriting and response parsing helpers
  ├── ingest.py          # Document loading, chunking, vector store, PDF extraction
  ├── requirements.txt


## Setup

```bash
# Clone the repo
git clone git@github.com:your-username/simple-rag-backend.git
cd simple-rag-backend

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Add your API keys
echo "ANTHROPIC_API_KEY=your-anthropic-key-here" >> .env
echo "API_SECRET_KEY=your-own-made-up-secret-here" >> .env
```

## Running the server

```bash
uvicorn main:app --reload
```

Visit `http://127.0.0.1:8000/docs` for interactive API documentation. Click **Authorize** and enter your `API_SECRET_KEY` to test protected endpoints directly from the docs UI.

## Authentication

All endpoints except `/` and `/health` require an API key, sent via the `X-API-Key` header:

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "X-API-Key: your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"question": "When did WWII start?"}'
```

## API Endpoints

### `POST /upload`
Add a text document to the knowledge base.

**Request:**
```json
{
  "source": "eiffel_tower.txt",
  "content": "The Eiffel Tower was completed in 1889..."
}
```

### `POST /upload-pdf`
Upload a PDF file (multipart form data). Text is extracted, chunked, and embedded automatically.

### `POST /query`
Ask a question. Supports multi-turn conversations via `conversation_id`.

**Request:**
```json
{
  "question": "When was the Eiffel Tower built?",
  "conversation_id": null
}
```

**Response:**
```json
{
  "answer": "The Eiffel Tower was completed in 1889.",
  "sources": ["eiffel_tower.txt"],
  "conversation_id": "generated-uuid-here",
  "search_query_used": "When was the Eiffel Tower built?"
}
```

Pass the returned `conversation_id` on subsequent requests to continue the same conversation — follow-up questions with pronouns or implicit references (e.g. "What about after that?") are automatically rewritten into standalone queries before retrieval.

### `POST /chat`
Direct, non-RAG chat with Claude (no document retrieval, no memory).

### `GET /health`
Health check endpoint (no auth required).

## Tech stack

- **FastAPI** — web framework and API layer
- **ChromaDB** — persistent vector database with built-in embeddings and HNSW-based similarity search
- **Anthropic Claude API** — response generation and query rewriting
- **NLTK** — sentence-boundary-aware text chunking
- **pypdf** — PDF text extraction

## Known limitations / possible next steps

- Conversation history is stored in-memory and resets on server restart — a production version would use Redis or a database
- No support for concurrent requests modifying the same conversation safely
- Chunk size/overlap are fixed constants rather than configurable per-document
- Could swap in a dedicated embedding model for more control over retrieval quality
