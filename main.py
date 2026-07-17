import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, File, UploadFile, Depends
from pypdf import PdfReader
from anthropic import Anthropic
from ingest import load_documents, build_vector_store, chunk_text
from models import ChatRequest, QueryRequest, UploadRequest
from auth import verify_api_key
from retrieval import rewrite_query, extract_text
import uuid, io

load_dotenv()   # read .env file

app = FastAPI() # Initialize the application
client = Anthropic() # pick up ANTHROPIC_API_KEY from env

# build the vector store once, when the server starts
docs = load_documents()
collection = build_vector_store(docs)

# in-memory store: conversation id -> list of {role, content} messages
conversation_store = {}

@app.get("/")
def read_root():
    return {"message": "Hello from your RAG API"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/chat")
def chat(request: ChatRequest, api_key: str = Depends(verify_api_key)):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    try:
        response = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=1024,        # required on every request, caps how long response can be
            messages=[{"role": "user", "content": request.message}]
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM request failed: {str(e)}")

    return {"response": extract_text(response)}

@app.post("/upload")
def upload_document(request: UploadRequest, api_key: str = Depends(verify_api_key)):
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
def query(request: QueryRequest, api_key: str = Depends(verify_api_key)):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    conversation_id = request.conversation_id or str(uuid.uuid4())
    history = conversation_store.get(conversation_id, [])

    # Rewrite the question to be self-contained before retrieval
    search_query = rewrite_query(client, request.question, history)

    results = collection.query(
        query_texts=[search_query],
        n_results=3
    )

    if not results["documents"][0]:
        raise HTTPException(status_code=404, detail="No relevant documents found")

    retrieved_chunks = results["documents"][0]
    context = "\n\n".join(retrieved_chunks)

    user_message = f"""Answer the question using only the context below. If the context doesn't contain the answer, say so.

Context:
{context}

Question: {request.question}"""

    messages = history + [{"role": "user", "content": user_message}]

    try:
        response = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=1024,
            messages=messages
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM request failed: {str(e)}")

    answer = extract_text(response)

    history.append({"role": "user", "content": request.question})
    history.append({"role": "assistant", "content": answer})
    conversation_store[conversation_id] = history

    return {
        "answer": answer,
        "sources": [m["source"] for m in results["metadatas"][0]],
        "conversation_id": conversation_id,
        "search_query_used": search_query  # exposed for transparency into the retrieval process, for demos
    }

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...), api_key: str = Depends(verify_api_key)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    pdf_bytes = await file.read()
    reader = PdfReader(io.BytesIO(pdf_bytes))

    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"

    if not text.strip():
        raise HTTPException(status_code=400, detail="Could not extract any text from this PDF")

    chunks = chunk_text(text)

    for i, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk],
            ids=[f"{file.filename}_{i}"],
            metadatas=[{"source": file.filename}]
        )

    return {
        "message": f"Uploaded {file.filename}",
        "chunks_added": len(chunks)
    }
