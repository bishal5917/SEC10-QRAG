# 🔍 Multimodal RAG — Text, Tables & Images from PDFs

A production-quality Retrieval-Augmented Generation system that answers questions from **text**, **tables**, and **images** extracted from PDF documents. Built with **LangChain** for document processing, **LangGraph** for workflow orchestration, and **Gemini** for multimodal generation.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INGESTION PIPELINE                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌──────────┐     ┌─────────────────┐     ┌─────────────────┐      │
│   │ PDF Files│────▶│  Document Proc.  │────▶│  Text Chunks    │──┐  │
│   │          │     │  • PyMuPDF text  │     │  Tables (MD)    │  │  │
│   │          │     │  • pdfplumber    │     │  Page Images    │  │  │
│   └──────────┘     │  • Page render   │     └─────────────────┘  │  │
│                     └─────────────────┘                          │  │
│                                                                  │  │
│                     ┌─────────────────┐                          │  │
│                     │   Embeddings    │◀─────────────────────────┘  │
│                     │  BGE (text/tbl) │                             │
│                     │  CLIP (images)  │                             │
│                     └────────┬────────┘                             │
│                              │                                      │
│                     ┌────────▼────────┐                             │
│                     │   Qdrant        │  (local persistent, hybrid) │
│                     │  3 collections  │                             │
│                     └─────────────────┘                             │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    QUERY WORKFLOW (LangGraph)                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌─────────┐    ┌──────────┐    ┌─────────┐    ┌──────────────┐    │
│   │ Retrieve│───▶│  Grade   │───▶│ Prepare │───▶│   Generate   │    │
│   │         │    │Documents │    │ Context │    │  (Gemini +   │    │
│   │ • Text  │    │          │    │         │    │   images)    │    │
│   │ • Table │    │(LLM-based│    │(filtered│    │              │    │
│   │ • Image │    │relevance)│    │ context)│    │  → Answer    │    │
│   └─────────┘    └──────────┘    └─────────┘    └──────────────┘    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Technology Stack

| Layer | Technology            | Purpose |
|-------|-----------------------|---------|
| **Orchestration** | LangGraph             | Stateful workflow graph (retrieve→grade→generate) |
| **Framework** | LangChain             | Document loaders, text splitters, embeddings interface |
| **Text Embeddings** | BGE-large-en-v1.5     | 1024-dim embeddings for text and table retrieval |
| **Image Embeddings** | OpenCLIP ViT-B-32     | 512-dim CLIP embeddings for visual retrieval |
| **Vector Store** | Qdrant (embedded, local) | Persistent storage + native hybrid search, no Docker required |
| **LLM** | Gemini 3.1 Flash Lite | Free-tier multimodal generation (text + images) |
| **PDF Processing** | PyMuPDF + pdfplumber  | Text extraction, table parsing, page rendering |
| **Configuration** | Pydantic Settings     | Type-safe .env-based configuration |

## Key Design Decisions

1. **LangGraph for orchestration** — Provides typed state management, clear node boundaries, and the ability to add conditional routing (e.g., query reformulation on low relevance).

2. **Separate embedding models** — BGE for text/tables (optimized for semantic text search) and CLIP for images (enables text→image cross-modal retrieval).

3. **Three Qdrant collections** — Independent retrieval per modality; text and tables use hybrid (dense + sparse) search, images use dense CLIP vectors.

4. **Document grading** — LLM-based relevance filtering between retrieval and generation prevents the LLM from being misled by noisy context.

5. **Multimodal generation** — Page images are sent directly to Gemini's vision input for chart/graph interpretation that text extraction alone cannot capture.

---

## Project Structure

```
TTIRAG/
├── app/                           # Main application package
│   ├── __init__.py
│   ├── core/                      # Configuration & utilities
│   │   ├── __init__.py
│   │   ├── config.py              # Pydantic Settings (paths, models, params)
│   │   └── logging.py            # Rich logging setup
│   ├── document_processing/       # PDF → Documents
│   │   ├── __init__.py
│   │   ├── pdf_loader.py          # PyMuPDF text extraction (LangChain loader)
│   │   ├── text_splitter.py       # RecursiveCharacterTextSplitter
│   │   ├── table_extractor.py     # pdfplumber → Markdown tables
│   │   └── image_extractor.py    # Page rendering to PNG
│   ├── embeddings/                # Embedding models
│   │   ├── __init__.py
│   │   ├── text_embeddings.py    # BGE-large (LangChain Embeddings)
│   │   └── clip_embeddings.py    # OpenCLIP ViT-B-32 (LangChain Embeddings)
│   ├── vectorstore/               # Qdrant storage
│   │   ├── __init__.py
│   │   └── store.py              # Multi-collection Qdrant manager (hybrid)
│   ├── retrieval/                 # Multi-modal retrieval
│   │   ├── __init__.py
│   │   └── multimodal_retriever.py  # Combined text+table+image retrieval
│   ├── chains/                    # LangChain chains
│   │   ├── __init__.py
│   │   ├── grader.py             # Document relevance grading
│   │   └── generation.py        # Gemini multimodal generation
│   └── graph/                     # LangGraph workflow
│       ├── __init__.py
│       ├── state.py              # TypedDict state schema
│       ├── nodes.py              # Graph node functions
│       └── workflow.py           # Graph construction & RAGWorkflow class
├── data/
│   └── pdfs/                     # Input PDF files
├── extracted/                     # (auto-generated) Rendered page images
├── vectorstore/                   # (auto-generated) Qdrant persistent data
├── ingest.py                      # CLI: Run ingestion pipeline
├── query.py                       # CLI: Ask questions (interactive/single)
├── requirements.txt               # All dependencies (pinned versions)
├── .env.example                   # API key template
└── .gitignore
```
---

## Usage

### Ingest PDFs

```bash
# Process all PDFs and build the vector store
python ingest.py

# Clear existing data and re-ingest from scratch
python ingest.py --clear

```

### Query

```bash
# Interactive mode (REPL)
python query.py

# Single question
python query.py -q "What was Apple's revenue in Q3 2023?"

```

---

## How the LangGraph Workflow Operates

```python
# Simplified view of the graph execution:

State["question"] = "What was AAPL's net income in Q3 2023?"
      │
      ▼
┌─ RETRIEVE ─────────────────────────────────────┐
│  • Embed query with BGE → search text/tables   │
│  • Embed query with CLIP → search images       │
│  → 5 text chunks + 3 tables + 2 images         │
└─────────────────────────────────────────────────┘
      │
      ▼
┌─ GRADE ────────────────────────────────────────┐
│  • LLM evaluates each doc: relevant/irrelevant │
│  → 3 text chunks + 2 tables pass              │
└─────────────────────────────────────────────────┘
      │
      ▼
┌─ PREPARE CONTEXT ──────────────────────────────┐
│  • Assemble graded docs + ungraded images      │
│  → Structured context ready for generation     │
└─────────────────────────────────────────────────┘
      │
      ▼
┌─ GENERATE ─────────────────────────────────────┐
│  • Send text context + page images to Gemini   │
│  → Multimodal answer with citations           │
└─────────────────────────────────────────────────┘
```

---