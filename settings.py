"""
settings.py
------------
Centralized configuration for the RAG pipeline.

WHY THIS FILE EXISTS:
In a real project you NEVER want to instantiate your embedding model
in five different files. If you ever switch from OpenAI embeddings to
a local HuggingFace model, you want to change it in ONE place.

This is the "embedding model class" referenced in the curriculum --
a single source of truth for which embedding model and LLM the whole
app uses.
"""

import os
from dotenv import load_dotenv

load_dotenv()  # reads .env file into environment variables

# ---- Choose your embedding provider ----
# Option A: OpenAI (needs OPENAI_API_KEY in .env, costs a few cents)
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

EMBEDDING_MODEL_NAME = "text-embedding-3-small"
LLM_MODEL_NAME = "gpt-4o-mini"

def get_embedding_model():
    """
    Returns a LangChain Embeddings object.
    This single function is called by BOTH:
      1. ingest.py   (to embed PDF chunks)
      2. rag_pipeline.py (to embed the user's question)
    Using the same function guarantees both sides use the
    same vector space -- this is mandatory for RAG to work.
    """
    return OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)


def get_llm():
    """Returns the chat LLM used for the generation step."""
    return ChatOpenAI(model=LLM_MODEL_NAME, temperature=0)


# ---- Other shared config ----
CHUNK_SIZE = 1000          # characters per chunk
CHUNK_OVERLAP = 150        # overlap between chunks to preserve context
VECTOR_DB_DIR = "chroma_db"  # folder where Chroma persists vectors
COLLECTION_NAME = "pdf_docs"
TOP_K = 6                  # how many chunks to retrieve per question
