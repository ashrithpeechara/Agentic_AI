"""
Verification test for Task 4 — RAG-Based Question Answering
Tests: document loading, chunking, embedding, retrieval, and answer generation.
"""

import sys, os
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv()

print("=" * 58)
print("   Task 4 -- RAG Q&A Verification Test")
print("=" * 58)

# ── Test 1: Document Loading ──────────────────────────────
print("\n[Test 1] Loading sample TXT document...")
from rag import load_document, chunk_text

sample_path = os.path.join(os.path.dirname(__file__), "sample.txt")
text = load_document(sample_path)
assert len(text) > 500, "Document too short"
word_count = len(text.split())
print(f"  PASS  |  {word_count} words loaded")

# ── Test 2: Chunking ──────────────────────────────────────
print("[Test 2] Chunking document...")
chunks = chunk_text(text, chunk_size=400, overlap=80)
assert len(chunks) >= 2, f"Expected multiple chunks, got {len(chunks)}"
assert all("text" in c and "id" in c for c in chunks), "Invalid chunk format"
print(f"  PASS  |  {len(chunks)} chunks created  |  avg {sum(c['word_count'] for c in chunks)//len(chunks)} words/chunk")

# ── Test 3: Cosine Similarity ─────────────────────────────
print("[Test 3] Cosine similarity (pure Python)...")
from rag import cosine_similarity
a = [1.0, 0.0, 0.0]
b = [1.0, 0.0, 0.0]
c = [0.0, 1.0, 0.0]
assert abs(cosine_similarity(a, b) - 1.0) < 1e-6,  "Identical vectors should be 1.0"
assert abs(cosine_similarity(a, c) - 0.0) < 1e-6,  "Orthogonal vectors should be 0.0"
print("  PASS  |  identical=1.0  |  orthogonal=0.0")

# ── Test 4: Embedding ─────────────────────────────────────
print("[Test 4] Embedding 2 test chunks (API call)...")
from rag import embed_text
emb1 = embed_text("machine learning algorithms")
emb2 = embed_text("neural network deep learning")
assert len(emb1) > 100, "Embedding too short"
assert len(emb1) == len(emb2), "Embeddings must be same dimension"
sim = cosine_similarity(emb1, emb2)
print(f"  PASS  |  dim={len(emb1)}  |  similarity(ML, DL)={sim:.3f}")

# ── Test 5: Retrieval ─────────────────────────────────────
print("[Test 5] Full RAG pipeline — load, chunk, embed, retrieve...")
from rag import RAGPipeline

rag = RAGPipeline()
stats = rag.load(sample_path)
assert rag.is_loaded, "Pipeline should be loaded"
print(f"  PASS  |  {stats['num_chunks']} chunks embedded")

# ── Test 6: Query & Answer ────────────────────────────────
print("[Test 6] RAG query — 'What is Retrieval-Augmented Generation?' ...")
result = rag.query("What is Retrieval-Augmented Generation?")
assert "answer" in result, "Result must have 'answer'"
assert "sources" in result, "Result must have 'sources'"
assert len(result["answer"]) > 50, "Answer too short"
assert len(result["sources"]) >= 1, "Must return at least 1 source"
print(f"  PASS  |  Answer: {result['answer'][:120]}...")
print(f"         |  Sources: {len(result['sources'])}  |  Top score: {result['sources'][0]['score']}")

# ── Test 7: Relevance check ───────────────────────────────
print("[Test 7] Verifying retrieved chunks are relevant...")
top_chunk = result["sources"][0]
assert top_chunk["score"] > 0.5, f"Expected score > 0.5, got {top_chunk['score']}"
assert "rag" in top_chunk["text"].lower() or "retrieval" in top_chunk["text"].lower(), \
    "Top chunk should mention RAG/retrieval"
print(f"  PASS  |  Top chunk score: {top_chunk['score']} (> 0.5)")

print("\n" + "=" * 58)
print("  All tests passed! Task 4 is working correctly.")
print("  Run  `python app.py`  to start the web UI (port 5003).")
print("=" * 58 + "\n")
