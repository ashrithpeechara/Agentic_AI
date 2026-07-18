"""
Task 4 – RAG Q&A: Core RAG Engine
====================================
Pipeline:
  1. Load PDF or TXT document
  2. Split into overlapping chunks
  3. Embed each chunk with Google text-embedding-004
  4. On query: embed query → cosine similarity → top-k chunks
  5. Pass retrieved context + question to Gemini for answer generation
"""

import os
import re
import sys
import math

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise EnvironmentError("GEMINI_API_KEY not set in .env file.")

client = genai.Client(api_key=API_KEY)

EMBED_MODEL = "gemini-embedding-001"
LLM_MODEL   = "gemini-3.5-flash"

# ─── Chunking config ──────────────────────────────────────
CHUNK_SIZE    = 400   # words per chunk
CHUNK_OVERLAP = 80    # words overlap between adjacent chunks


# ─────────────────────────────────────────────────────────────
# 1. Document Loading
# ─────────────────────────────────────────────────────────────
def load_txt(path: str) -> str:
    """Load a plain text file."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def load_pdf(path: str) -> str:
    """Load text from a PDF using pypdf."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(path)
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text.strip())
        return "\n\n".join(pages)
    except ImportError:
        raise ImportError("pypdf not installed. Run: pip install pypdf")


def load_document(path: str) -> str:
    """Load PDF or TXT document and return raw text."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return load_pdf(path)
    elif ext in {".txt", ".md", ".text"}:
        return load_txt(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Use PDF or TXT.")


# ─────────────────────────────────────────────────────────────
# 2. Text Chunking
# ─────────────────────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """
    Split text into overlapping word-based chunks.
    Returns list of {id, text, word_count, start_word} dicts.
    """
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()

    chunks = []
    start = 0
    chunk_id = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunk_text  = " ".join(chunk_words)

        chunks.append({
            "id":         chunk_id,
            "text":       chunk_text,
            "word_count": len(chunk_words),
            "start_word": start,
        })

        chunk_id += 1
        if end == len(words):
            break
        start += chunk_size - overlap  # slide forward with overlap

    return chunks


# ─────────────────────────────────────────────────────────────
# 3. Embedding
# ─────────────────────────────────────────────────────────────
def embed_text(text: str) -> list[float]:
    """Get embedding vector for a single text string."""
    response = client.models.embed_content(
        model=EMBED_MODEL,
        contents=text,
    )
    return response.embeddings[0].values


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Add 'embedding' field to each chunk.
    Embeds in small batches to respect rate limits.
    """
    BATCH = 5
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i : i + BATCH]
        for chunk in batch:
            chunk["embedding"] = embed_text(chunk["text"])
    return chunks


# ─────────────────────────────────────────────────────────────
# 4. Similarity Search (Cosine)
# ─────────────────────────────────────────────────────────────
def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors (pure Python)."""
    dot   = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def retrieve(query: str, chunks: list[dict], top_k: int = 4) -> list[dict]:
    """
    Embed the query and return the top-k most similar chunks.
    Each returned chunk gets a 'score' field.
    """
    query_emb = embed_text(query)

    scored = []
    for chunk in chunks:
        score = cosine_similarity(query_emb, chunk["embedding"])
        scored.append({**chunk, "score": round(score, 4)})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


# ─────────────────────────────────────────────────────────────
# 5. Answer Generation
# ─────────────────────────────────────────────────────────────
def generate_answer(query: str, retrieved_chunks: list[dict]) -> dict:
    """
    Use retrieved chunks as context and Gemini to generate an answer.
    Returns {answer, sources, confidence}.
    """
    # Build context block
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        context_parts.append(
            f"[Source {i} | relevance: {chunk['score']:.2f}]\n{chunk['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""You are a precise question-answering assistant.
Answer the question ONLY using the provided context.
If the answer is not in the context, say "I couldn't find that information in the document."
Be specific, cite which source number supports your answer, and be concise.

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""

    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=(
                "You are an expert document analyst. Answer questions accurately "
                "based solely on the provided context. Always mention the source "
                "number that supports your answer."
            ),
            temperature=0.1,
        ),
    )

    return {
        "answer":  response.text.strip(),
        "sources": [
            {
                "id":    c["id"],
                "text":  c["text"][:300] + ("..." if len(c["text"]) > 300 else ""),
                "score": c["score"],
            }
            for c in retrieved_chunks
        ],
    }


# ─────────────────────────────────────────────────────────────
# Full RAG Pipeline (for CLI use)
# ─────────────────────────────────────────────────────────────
class RAGPipeline:
    """Encapsulates the full RAG pipeline state."""

    def __init__(self):
        self.chunks:   list[dict] = []
        self.doc_name: str = ""
        self.doc_text: str = ""

    def load(self, path: str) -> dict:
        """Load, chunk, and embed a document. Returns stats."""
        print(f"  Loading document: {path}")
        self.doc_text = load_document(path)
        self.doc_name = os.path.basename(path)

        print(f"  Chunking text ({len(self.doc_text.split())} words)...")
        raw_chunks = chunk_text(self.doc_text)

        print(f"  Embedding {len(raw_chunks)} chunks...")
        self.chunks = embed_chunks(raw_chunks)

        return {
            "doc_name":    self.doc_name,
            "total_words": len(self.doc_text.split()),
            "num_chunks":  len(self.chunks),
        }

    def query(self, question: str, top_k: int = 4) -> dict:
        """Run a RAG query against the loaded document."""
        if not self.chunks:
            return {"error": "No document loaded. Call load() first."}

        retrieved = retrieve(question, self.chunks, top_k=top_k)
        return generate_answer(question, retrieved)

    @property
    def is_loaded(self) -> bool:
        return len(self.chunks) > 0


# ─────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────
def main():
    DIVIDER = "=" * 60
    print(f"\n{DIVIDER}")
    print("  Task 4 -- RAG-Based Question Answering")
    print(DIVIDER)

    import sys
    if len(sys.argv) > 1:
        doc_path = sys.argv[1]
    else:
        sample = os.path.join(os.path.dirname(__file__), "sample.txt")
        doc_path = input(f"Document path (Enter for sample.txt): ").strip() or sample

    rag = RAGPipeline()
    stats = rag.load(doc_path)
    print(f"\n  Document loaded: {stats['doc_name']}")
    print(f"  Words: {stats['total_words']}  |  Chunks: {stats['num_chunks']}")
    print(f"\n  Ask questions about the document. Type 'quit' to exit.\n")

    while True:
        q = input("Question: ").strip()
        if not q:
            continue
        if q.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break

        result = rag.query(q)
        print(f"\n{'─'*60}")
        print(f"Answer:\n{result['answer']}")
        print(f"\nTop sources used:")
        for s in result["sources"][:2]:
            print(f"  [Chunk {s['id']} | score: {s['score']}] {s['text'][:100]}...")
        print(f"{'─'*60}\n")


if __name__ == "__main__":
    main()
