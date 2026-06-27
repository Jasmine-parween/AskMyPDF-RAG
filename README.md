# AskMyPDF — RAG-Powered PDF Q&A

A minimal, end-to-end **Retrieval-Augmented Generation (RAG)** pipeline that lets you chat with any PDF. Ask questions in plain English and get answers grounded in the document — with source page references so you can verify every response.

Built with **LangChain · ChromaDB · OpenAI · Streamlit**.

---

## How it works

```
Phase 1 — Indexing (run once per PDF)
──────────────────────────────────────
PDF file ──► Text splitter ──► Embedding model ──► ChromaDB (local vector store)

Phase 2 — Q&A (every question)
───────────────────────────────
Your question ──► Embed question ──► Retrieve top-k similar chunks
                                           │
                                           ▼
                              Augmented prompt (context + question)
                                           │
                                           ▼
                                    GPT-4o-mini ──► Grounded answer + sources
```

The pipeline always includes a **document overview chunk** (title + abstract) regardless of query similarity scores, which ensures general questions like *"What is this PDF about?"* and *"Summarise this document"* always return a useful answer.

---

## Prerequisites

- Python 3.9+
- An [OpenAI API key](https://platform.openai.com/api-keys)

---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/AskMyPDF.git
cd AskMyPDF

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your API key
cp .env.example .env
# Open .env and replace the placeholder with your actual key
```

---

## Usage

### Step 1 — Ingest your PDF

Put your PDF in the `data/` folder and update `PDF_PATH` at the top of [ingest.py](ingest.py) if needed, then run:

```bash
python ingest.py
```

This loads the PDF, splits it into overlapping chunks, embeds them with `text-embedding-3-small`, and persists the vectors to `chroma_db/`.

> Re-running `ingest.py` automatically wipes the old vector DB first, so you always get a clean index.

### Step 2 — Ask questions

**Option A — Streamlit UI (recommended)**

```bash
streamlit run app.py
```

Open the URL shown in your terminal, upload a PDF (or use the pre-ingested one), and start asking questions. Expand *"View source chunk(s) used"* under each answer to see exactly which pages the model drew from.

**Option B — Command line**

```bash
python rag_pipeline.py
```

Runs a quick test question against the pre-ingested PDF and prints the answer + source pages.

---

## Project structure

```
AskMyPDF/
├── data/               # Put your PDF(s) here
├── chroma_db/          # Auto-generated vector store (git-ignored)
├── venv/               # Virtual environment (git-ignored)
│
├── settings.py         # Single source of truth: models, chunk size, TOP_K
├── ingest.py           # Phase 1 — load → split → embed → persist
├── rag_pipeline.py     # Phase 2 — retrieve → prompt → LLM → answer
├── app.py              # Streamlit UI wrapping the same pipeline
│
├── requirements.txt
├── .env.example        # Copy to .env and add your API key
└── .env                # Your real key (never committed)
```

| File | What it does |
|---|---|
| `settings.py` | Centralises all config — swap models or tune chunk size here |
| `ingest.py` | One-time indexing; re-run whenever your PDF changes |
| `rag_pipeline.py` | The retrieval + generation chain (LangChain LCEL) |
| `app.py` | Interactive Streamlit frontend |

---

## Configuration

All tunable parameters live in [settings.py](settings.py):

| Parameter | Default | Effect |
|---|---|---|
| `EMBEDDING_MODEL_NAME` | `text-embedding-3-small` | OpenAI embedding model |
| `LLM_MODEL_NAME` | `gpt-4o-mini` | Chat model for generation |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `150` | Overlap between adjacent chunks |
| `TOP_K` | `6` | Number of chunks retrieved per question |

---

## Things to try next

1. **Free local embeddings** — swap `OpenAIEmbeddings` for `HuggingFaceEmbeddings` in `settings.py` to run at zero cost.
2. **Multi-PDF support** — embed several files into the same Chroma collection, tagging each chunk with its filename.
3. **Conversational memory** — add `ConversationBufferMemory` so follow-up questions reference earlier answers.
4. **Cloud vector DB** — replace Chroma with Pinecone or Qdrant for a production-ready deployment.
5. **Tune retrieval** — experiment with `CHUNK_SIZE`, `CHUNK_OVERLAP`, and `TOP_K` and observe how answer quality changes.

---

## Tech stack

| Component | Library / Service |
|---|---|
| Orchestration | LangChain (LCEL) |
| Embeddings | OpenAI `text-embedding-3-small` |
| LLM | OpenAI `gpt-4o-mini` |
| Vector store | ChromaDB (local, file-based) |
| PDF loading | PyPDFLoader |
| UI | Streamlit |
