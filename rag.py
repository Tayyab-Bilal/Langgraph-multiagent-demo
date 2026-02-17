import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

POLICY_DOCS_DIR = "policy_docs"

def build_vectorstore():
    """Load all policy docs and build FAISS vector store."""
    docs = []
    
    for filename in os.listdir(POLICY_DOCS_DIR):
        if filename.endswith((".txt", ".md")):
            path = os.path.join(POLICY_DOCS_DIR, filename)
            loader = TextLoader(path, encoding="utf-8")
            docs.extend(loader.load())
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    print(f"RAG ready: {len(chunks)} chunks from {len(docs)} policy docs")
    return vectorstore


def retrieve_context(vectorstore, query: str, k: int = 3) -> str:
    """Search the vectorstore and return relevant policy context as a string."""
    results = vectorstore.similarity_search(query, k=k)
    
    if not results:
        return "No relevant policy information found."
    
    context_parts = []
    for i, doc in enumerate(results, 1):
        source = os.path.basename(doc.metadata.get("source", "unknown"))
        context_parts.append(f"[{i}] From {source}:\n{doc.page_content}")
    
    return "\n\n".join(context_parts)

_vectorstore = None
def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = build_vectorstore()
    return _vectorstore

