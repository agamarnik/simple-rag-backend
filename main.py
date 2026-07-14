import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from anthropic import Anthropic
from ingest import load_documents, build_vector_store, chunk_text
from typing import Optional
import uuid

load_dotenv()   # read .env file

app = FastAPI() # Initialize the application
client = Anthropic() # pick up ANTHROPIC_API_KEY from env

# build the vector store once, when the server starts
docs = load_documents()
collection = build_vector_store(docs)

# in-memory store: conversation id -> list of {role, content} messages
conversation_store = {}

class ChatRequest(BaseModel):
    message: str

class QueryRequest(BaseModel):
    question: str
    conversation_id: Optional[str] = None

class UploadRequest(BaseModel):
    source: str         # file name/label
    content: str        # file content

@app.get("/")
def read_root():
    return {"message": "Hello from your RAG API"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/chat")
def chat(request: ChatRequest):
    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=1024, # required on every request, caps how long response can be
        messages=[
            {"role": "user", "content": request.message}
        ]
    )
    return {"response": response.content[0].text}

@app.post("/upload")
def upload_document(request: UploadRequest):
    if not request.content.strip():     # error handling for empty doc
        raise HTTPException(status_code=400, detail="Content cannot be empty")

    chunks = chunk_text(request.content)

    for i, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk],
            ids=[f"{request.source}_{i}"],
            metadatas=[{"source": request.source}]
        )

    return {
        "message": f"Uploaded {request.source}",
        "chunks_added": len(chunks)
    }

@app.post("/query")
def query(request: QueryRequest):
    if not request.question.strip():        # error handling for empty query
        raise HTTPException(status_code=400, detail="Question cannot be empty") # status_code 400: client send something invalid

    # 1. Retrieve relevant chunks
    results = collection.query(
        query_texts=[request.question],
        n_results=3
    )

    if not results["documents"][0]:
        raise HTTPException(status_code=404, detail="No relevant documents found") # status_code 404: nothing found

    retrieved_chunks = results["documents"][0]
    context = "\n\n".join(retrieved_chunks)

    # Get or start conversation history
    conversation_id = request.conversation_id or str(uuid.uuid4())
    history = conversation_store.get(conversation_id, [])

    # 2. Build the current user turn, including retrieved context
    # "using only the context below" is what makes answers grounded in the specific docs rather than generic Claude knowledge
    user_message = f"""Answer the question using only the context below. If the context doesn't contain the answer, say so.

Context:
{context}

Question: {request.question}"""

    messages = history + [{"role": "user", "content": user_message}]

    # 3. Send to Claude
    try:
        response = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=1024,
            messages=messages
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM request failed: {str(e)}")    # status_code 502: Bad gateway, Claude failed


    answer = response.content[0].text

    # Save this turn to  history (store the raw question, not the context-stuffed version)
    history.append({"role": "user", "content": request.question})
    history.append({"role": "assistant", "content": answer})
    conversation_store[conversation_id] = history

    return {
        "answer": answer,
        "sources": [m["source"] for m in results["metadatas"][0]],
        "conversation_id": conversation_id
    }
