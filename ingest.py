"""
ingest.py
---------
PHASE 1 of the RAG pipeline: turn a PDF into a searchable vector store.
 
Run this ONCE (or whenever your PDF changes):
    python ingest.py
 
Pipeline steps (matches the architecture diagram):
    PDF file -> Text splitter -> Embedding model -> Vector DB
"""
 
import os
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
 
from langchain_core.prompts import ChatPromptTemplate
 
from settings import (
    get_embedding_model,
    get_llm,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    VECTOR_DB_DIR,
    COLLECTION_NAME,
)
 
PDF_PATH = "data/2606.26028v1.pdf"  # change this to your file name
 
 
def load_pdf(path: str):
    """Step 1: Load the PDF. Each page becomes one LangChain Document."""
    print(f"[1/4] Loading PDF from {path} ...")
    loader = PyPDFLoader(path)
    documents = loader.load()
    print(f"      Loaded {len(documents)} pages.")
    return documents
 
 
def build_overview_chunk(documents, llm):
    """
    Creates a synthetic 'Document Overview' chunk using an LLM-generated summary.
 
    Why: generic queries like 'what is this about?' or 'summarise the PDF' use
    conversational language that doesn't embed close to technical paper content.
    Without this chunk those queries never retrieve the abstract, so the LLM
    correctly but frustratingly says 'I don't have enough information'.
    Using an LLM summary (rather than blindly taking pages 1-2) makes this
    robust to any document type -- invoices, manuals, legal contracts, etc.
    """
    sample_text = "\n\n".join(d.page_content for d in documents[:5])
    prompt = ChatPromptTemplate.from_template(
        "Write a 4-sentence overview of the following document. "
        "Cover: what the document is, its main purpose, key topics, and intended audience.\n\n"
        "{text}"
    )
    summary = (prompt | llm).invoke({"text": sample_text}).content
    print(f"      Overview summary generated ({len(summary)} chars).")
    return Document(
        page_content=summary,
        metadata={"source": "document_overview", "page": 0},
    )
 
 
def split_documents(documents):
    """Step 2: Split pages into smaller chunks."""
    print(f"[2/4] Splitting into chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}) ...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    print(f"      Created {len(chunks)} chunks.")
    return chunks
 
 
def embed_and_store(chunks):
    """Step 3 & 4: Embed each chunk and store the vectors in Chroma."""
    print("[3/4] Loading embedding model ...")
    embedding_model = get_embedding_model()
 
    print("[4/4] Embedding chunks and writing to vector DB ...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=VECTOR_DB_DIR,
        collection_name=COLLECTION_NAME,
    )
    print(f"      Done. Vector DB persisted at ./{VECTOR_DB_DIR}/")
    return vectorstore
 
 
if __name__ == "__main__":
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(
            f"Could not find {PDF_PATH}. Put a PDF in the data/ folder "
            f"and update PDF_PATH at the top of ingest.py."
        )
 
    # Wipe the old DB so we start clean (avoids duplicate chunks on re-ingest)
    if os.path.exists(VECTOR_DB_DIR):
        print(f"Removing existing vector DB at ./{VECTOR_DB_DIR}/ ...")
        shutil.rmtree(VECTOR_DB_DIR)
 
    docs = load_pdf(PDF_PATH)
 
    # Prepend a document-overview chunk before splitting
    print("[0/4] Generating document overview summary ...")
    llm = get_llm()
    overview = build_overview_chunk(docs, llm)
    chunks = split_documents(docs)
    # The overview chunk is large on purpose -- keep it whole, don't split it
    all_chunks = [overview] + chunks
 
    print(f"      Total chunks including overview: {len(all_chunks)}")
    embed_and_store(all_chunks)
    print("\nIngestion complete! You can now run rag_pipeline.py or app.py")