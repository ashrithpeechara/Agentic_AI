import os
from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def load_document(file_path):
    if file_path.lower().endswith(".pdf"):
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

def chunk_text(text, chunk_size=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

def retrieve_relevant_chunks(query, chunks, top_k=3):
    query_words = set(query.lower().split())
    scored_chunks = []
    for chunk in chunks:
        chunk_words = set(chunk.lower().split())
        score = len(query_words & chunk_words)
        scored_chunks.append((score, chunk))
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    return [chunk for score, chunk in scored_chunks[:top_k]]

def answer_query(query, context_chunks):
    context = "\n\n".join(context_chunks)
    prompt = (
        f"Answer the question using only the context below.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n"
        "Answer:"
    )
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    file_path = input("Enter path to PDF/TXT document: ")
    document_text = load_document(file_path)
    chunks = chunk_text(document_text)
    query = input("Enter your question: ")
    relevant_chunks = retrieve_relevant_chunks(query, chunks)
    answer = answer_query(query, relevant_chunks)
    print("\nAnswer:\n", answer)