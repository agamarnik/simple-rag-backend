from pathlib import Path
import chromadb
import nltk
nltk.download('punkt_tab')  # one-time download of the sentence tokenizer model
from nltk.tokenize import sent_tokenize

# search the docs/ folder and returns every file matching *.txt
# For each matching file, open it, read the whole thing into a string, and store it as a
# dictionary with two keys: source (just the filename) and content (the full text).
# source is kept so later user can know which doc the content came from
# returns: list of these dictionaries eg [ {"source": "sharks.txt", "content": "Sharks are..."}, {}, ... ]
def load_documents(folder="docs"):
    docs = []
    for file_path in Path(folder).glob("*.txt"):
        with open(file_path, "r") as f:
            docs.append({"source": file_path.name, "content": f.read()})
    return docs

# This slices one long string into overlapping sentences.
def chunk_text(text, max_chunk_size=500, overlap_sentences=1):
    sentences = sent_tokenize(text)     # split text into sentences
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        current_chunk.append(sentence)
        current_length += len(sentence)

        if current_length >= max_chunk_size:
            chunks.append(" ".join(current_chunk))
            # keep the last N sentences as overlap into the next chunk
            current_chunk = current_chunk[-overlap_sentences:]
            current_length = sum(len(s) for s in current_chunk)

    if current_chunk:       # saves the final chunk
        chunks.append(" ".join(current_chunk))

    return chunks

def build_vector_store(docs):
    client = chromadb.PersistentClient(path="./chroma_db")       # persistent vector store survives server restarts
    collection = client.get_or_create_collection(name="my_docs") # creates or reuse one table/bucket of vectors (aka collection), called "my_docs"

    if collection.count() == 0:                 # embed docs first time
        for doc in docs:                        # for each doc, chunk it
            chunks = chunk_text(doc["content"])
            for i, chunk in enumerate(chunks):  # for each chunk, call collection.add(...)
                collection.add(                 # this is where the chroma runs the embedding model on each chunk and stores resulting vector
                    documents=[chunk],
                    ids=[f"{doc['source']}_{i}"],
                    metadatas=[{"source": doc["source"]}]   # extra info stored alongside vector
                )
    else:                                       # reuse already embedded docs
        print(f"Using existing collection with {collection.count()} chunks")

    return collection

def query_collection(collection, question, n_results=3):
    results = collection.query(
        query_texts=[question],
        n_results=n_results
    )
    return results

if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    collection = build_vector_store(docs)
    print(f"Collection has {collection.count()} chunks")

    # Test query, based on existing WW2 doc
    question = "When was world war 2?"
    results = query_collection(collection, question)

    for i, (doc, metadata, distance) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    )):
        print(f"\n--- Result {i+1} (source: {metadata['source']}, distance: {distance:.4f}) ---")
        print(doc[:200])  # first 200 chars of the chunk
