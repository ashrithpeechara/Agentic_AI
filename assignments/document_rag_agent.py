from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

def create_vector_store(pdf_path):

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vector_store


def ask_document(vector_store, question, llm):

    docs = vector_store.similarity_search(
        question,
        k=4
    )

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    prompt = f"""
You are a document analysis agent.

Answer the user's question using ONLY the provided
document context.

If the answer cannot be found in the context,
say that the information is not available.

DOCUMENT CONTEXT:
{context}

QUESTION:
{question}

Provide a clear and concise answer.
"""

    response = llm.invoke(prompt)

    return response.content

vector_store = create_vector_store(
    "data/documents/sample.pdf"
)

answer = ask_document(
    vector_store,
    "What is the main purpose of this document?",
    llm
)

print(answer)