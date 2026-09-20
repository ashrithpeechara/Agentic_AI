import os
import math
import time
from dotenv import load_dotenv
from google import genai
from google.genai import errors

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

def load_documents(doc_dir="documents"):
    docs = []
    if not os.path.exists(doc_dir):
        os.makedirs(doc_dir, exist_ok=True)
        sample_path = os.path.join(doc_dir, "agentic_ai.txt")
        with open(sample_path, "w", encoding="utf-8") as f:
            f.write("""Agentic AI refers to autonomous artificial intelligence systems designed to pursue complex goals with minimal human intervention.
Core components of an Agentic AI system include:
1. Perception and Environment Interface: Gathering signals and context from tools, APIs, and databases.
2. Memory Systems: Short-term context memory and long-term vector storage for recalling facts and previous interactions.
3. Planning and Reasoning: Decomposing high-level tasks into sub-goals using techniques like ReAct (Reason + Act) and Chain-of-Thought.
4. Tool Execution: Autonomous calling of calculators, search engines, databases, and external web APIs.
5. Self-Reflection: Evaluating outcomes, catching errors, and revising strategies iteratively.

Traditional AI vs Agentic AI:
Traditional AI systems are passive, request-response models that produce static outputs from single prompts. In contrast, Agentic AI acts proactively, maintains goal state, executes external actions in loops, and corrects course when obstacles arise.""")
    
    for fname in os.listdir(doc_dir):
        if fname.endswith(".txt"):
            with open(os.path.join(doc_dir, fname), "r", encoding="utf-8") as f:
                docs.append(f.read())
    return docs

def chunk_text(text, chunk_size=300):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - 40):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

def embed_text(text):
    resp = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )
    return resp.embeddings[0].values

def safe_generate(prompt):
    for attempt in range(3):
        try:
            return client.models.generate_content(
                model="gemini-3.5-flash",
                contents=prompt
            ).text.strip()
        except errors.ClientError as e:
            if "429" in str(e):
                time.sleep(10)
            else:
                raise

def run_rag_system(question: str):
    docs = load_documents("documents")
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_text(doc))
    
    chunk_embeddings = [embed_text(c) for c in all_chunks]
    q_emb = embed_text(question)
    
    scored = []
    for chunk, c_emb in zip(all_chunks, chunk_embeddings):
        sim = cosine_similarity(q_emb, c_emb)
        scored.append((sim, chunk))
    scored.sort(key=lambda x: x[0], reverse=True)
    top_chunks = [c for _, c in scored[:2]]
    context = "\n\n".join(top_chunks)
    
    prompt = f"""Use the following context to answer the user's question accurately.
If the answer cannot be found in the context, say you don't know.

Context:
{context}

Question: {question}

Answer:"""
    
    answer = safe_generate(prompt)
    print(f"\n[Question]: {question}")
    print(f"[Answer]: {answer}")

if __name__ == "__main__":
    print("--- Running Task 2: RAG-Based QA System ---")
    run_rag_system("What are the core components of an Agentic AI?")
    run_rag_system("How does Agentic AI differ from traditional AI?")
