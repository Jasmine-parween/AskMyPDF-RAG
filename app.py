"""
app.py
------
Streamlit UI for the RAG PDF workflow.

Run with:
    streamlit run app.py

This UI lets a user:
  1. Upload a PDF (optional -- or use the pre-ingested one)
  2. Ask questions about it
  3. See the answer + which page(s) it came from (source traceability)
"""

import os
import tempfile
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from settings import (
    get_embedding_model,
    get_llm,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)
from rag_pipeline import PROMPT_TEMPLATE, format_docs, make_retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


st.set_page_config(page_title="RAG PDF Q&A", page_icon="📄", layout="centered")
st.title("📄 Chat with your PDF (RAG)")
st.caption("Upload a PDF, then ask questions about it. Answers are grounded in the document.")


# ---------- Session state ----------
# Streamlit re-runs the whole script on every interaction, so we cache
# the vectorstore in session_state to avoid re-embedding on every click.
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of (question, answer, sources)


# ---------- Step 1: Upload + ingest ----------
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None:
    if st.button("Process PDF", type="primary"):
        with st.spinner("Reading PDF, splitting into chunks, and embedding..."):
            # Save uploaded file to a temp path so PyPDFLoader can read it
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            # Phase 1: load -> split -> embed -> store (in-memory Chroma for this session)
            loader = PyPDFLoader(tmp_path)
            documents = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
            )
            chunks = splitter.split_documents(documents)

            embedding_model = get_embedding_model()
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embedding_model,
                collection_name="streamlit_session",
            )

            st.session_state.vectorstore = vectorstore
            st.session_state.chat_history = []  # reset chat for new doc
            os.unlink(tmp_path)

        st.success(f"Processed! {len(chunks)} chunks indexed from {len(documents)} pages.")


# ---------- Step 2: Ask questions ----------
if st.session_state.vectorstore is not None:
    st.divider()
    question = st.text_input("Ask a question about the PDF:")

    if st.button("Ask") and question.strip():
        with st.spinner("Searching document and generating answer..."):
            retriever = make_retriever(st.session_state.vectorstore)
            prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
            llm = get_llm()

            rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

            answer = rag_chain.invoke(question)
            sources = retriever.invoke(question)

            st.session_state.chat_history.append((question, answer, sources))

    # ---------- Step 3: Show chat history (most recent first) ----------
    for q, a, sources in reversed(st.session_state.chat_history):
        st.markdown(f"**You:** {q}")
        st.markdown(f"**Answer:** {a}")
        with st.expander(f"View {len(sources)} source chunk(s) used"):
            for s in sources:
                page = s.metadata.get("page", "?")
                st.markdown(f"**Page {page}:**")
                st.text(s.page_content[:400] + ("..." if len(s.page_content) > 400 else ""))
        st.divider()
else:
    st.info("Upload a PDF and click 'Process PDF' to get started.")
