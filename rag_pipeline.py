"""
rag_pipeline.py
----------------
PHASE 2 of the RAG pipeline: answer a user's question using the PDF.

Pipeline steps (matches the architecture diagram):
    User question -> Embed question -> Retrieve top-k chunks
    -> Build augmented prompt -> LLM -> Answer

Run a quick test directly:
    python rag_pipeline.py
"""

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from settings import get_embedding_model, get_llm, VECTOR_DB_DIR, COLLECTION_NAME, TOP_K


PROMPT_TEMPLATE = """You are a helpful assistant answering questions about a document.
Use the context chunks below to answer the question. Synthesize across all chunks as needed.
If the context genuinely does not contain enough information to give a useful answer, say "I don't have enough information in the document to answer that."

Context:
{context}

Question: {question}

Answer:"""


def load_vectorstore():
    """Reconnect to the persisted Chroma vector DB created by ingest.py."""
    embedding_model = get_embedding_model()
    vectorstore = Chroma(
        persist_directory=VECTOR_DB_DIR,
        embedding_function=embedding_model,
        collection_name=COLLECTION_NAME,
    )
    return vectorstore


def format_docs(docs):
    """Join retrieved chunks into one context string, with page numbers for traceability."""
    formatted = []
    for d in docs:
        page = d.metadata.get("page", "?")
        formatted.append(f"[Page {page}]\n{d.page_content}")
    return "\n\n---\n\n".join(formatted)


def make_retriever(vectorstore):
    """
    Returns a RunnableLambda retriever that always prepends the document
    overview chunk, then fills remaining slots with MMR results.

    Why: meta-queries like "summarise the pdf" or "what is this about?" use
    conversational vocabulary that doesn't embed close to technical abstract
    text, so the overview chunk is never retrieved by pure similarity search.
    Pinning it guarantees the LLM always has the document context it needs.
    """
    base = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": TOP_K - 1, "fetch_k": 20},
    )

    raw = vectorstore._collection.get(
        where={"source": "document_overview"},
        include=["documents", "metadatas"],
    )
    overview_docs = []
    if raw["documents"]:
        overview_docs = [
            Document(page_content=raw["documents"][0], metadata=raw["metadatas"][0])
        ]

    def retrieve(query):
        return overview_docs + base.invoke(query)

    return RunnableLambda(retrieve)


def build_rag_chain():
    """
    Builds the full RAG chain using LangChain's LCEL (LangChain Expression Language).

    Read it like a recipe, top to bottom:
      1. retriever       -> fetches top-k similar chunks for the question
      2. format_docs     -> turns chunks into a single context string
      3. prompt          -> inserts {context} and {question} into the template
      4. llm             -> generates the answer
      5. StrOutputParser -> extracts plain text from the LLM response
    """
    vectorstore = load_vectorstore()
    retriever = make_retriever(vectorstore)

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    llm = get_llm()

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain, retriever


def ask(question: str):
    """Convenience function: ask a single question, get the answer + sources."""
    rag_chain, retriever = build_rag_chain()
    answer = rag_chain.invoke(question)
    sources = retriever.invoke(question)
    return answer, sources


if __name__ == "__main__":
    test_question = "What is this document about?"
    print(f"Q: {test_question}\n")

    answer, sources = ask(test_question)

    print(f"A: {answer}\n")
    print("Sources used:")
    for s in sources:
        page = s.metadata.get("page", "?")
        snippet = s.page_content[:100].replace("\n", " ")
        print(f"  - Page {page}: {snippet}...")
