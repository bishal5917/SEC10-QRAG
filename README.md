# Document Retriever — Multimodal RAG System

RAG-based information retrieval over SEC 10-Q PDF filings. Answers questions using text, tables, and figures/charts across multiple documents. Fully Dockerized with Ollama (no external API keys).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INGESTION PIPELINE                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  PDF Files (data/pdfs/)                                             │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────────────────────────────────────────────┐       │
│  │  pdf_loader.py — Per-page multimodal extraction          │       │
│  │                                                          │       │
│  │  Text Detection:                                         │       │
│  │    pymupdf get_text("text", sort=True)                   │       │
│  │                                                          │       │
│  │  Table Detection:                                        │       │
│  │    pdfplumber find_tables() → markdown with headers      │       │
│  │                                                          │       │
│  │  Figure Detection (2 methods):                           │       │
│  │    1. Raster images: pymupdf get_images() (>200x200px)   │       │
│  │    2. Vector charts: text-density analysis (<25% coverage)│       │
│  │                                                          │       │
│  │  Figure Description:                                     │       │
│  │    Render page at 150 DPI → send to llava → text desc    │       │
│  └─────────────────────────────────────────────────────────┘       │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────────────────────────────────────────────┐       │
│  │  vector_store.py — Chunking + Embedding + Storage        │       │
│  │                                                          │       │
│  │  Chunking:                                               │       │
│  │    Text → split at 800 chars with 150 overlap            │       │
│  │    Tables → kept whole (not split)                       │       │
│  │    Figures → kept whole (description as text)            │       │
│  │                                                          │       │
│  │  Embedding:                                              │       │
│  │    nomic-embed-text via Ollama (768 dims)                │       │
│  │                                                          │       │
│  │  Storage:                                                │       │
│  │    ChromaDB (persistent, cosine similarity)              │       │
│  │    Figure images saved to disk (not in DB metadata)      │       │
│  └─────────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                         RETRIEVAL PIPELINE                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  User Query                                                         │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────────────────────────────────────────────┐       │
│  │  retriever.py — Hybrid Search                            │       │
│  │                                                          │       │
│  │  1. Vector Search: ChromaDB cosine similarity (top 20)   │       │
│  │  2. BM25 Keyword Search: term frequency scoring (top 20) │       │
│  │  3. Reciprocal Rank Fusion: merge both (0.6/0.4 weight)  │       │
│  └─────────────────────────────────────────────────────────┘       │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────────────────────────────────────────────┐       │
│  │  reranker.py — LLM-as-Judge Reranking                    │       │
│  │                                                          │       │
│  │  Score each chunk 0-10 for relevance using llama3        │       │
│  │  20 candidates → top 8 kept                              │       │
│  └─────────────────────────────────────────────────────────┘       │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────────────────────────────────────────────┐       │
│  │  llm.py — Dual-Path Generation                           │       │
│  │                                                          │       │
│  │  Text/Table chunks → llama3:instruct (single call)       │       │
│  │  Figure chunks → llava + raw PNG image (per-figure call) │       │
│  │                                                          │       │
│  │  Final answer with source citations                      │       │
│  └─────────────────────────────────────────────────────────┘       │
│       │                                                             │
│       ▼                                                             │
│  Answer + Sources + Chunk Previews                                  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Component | Tool | Purpose |
|-----------|------|---------|
| LLM (text/tables) | `llama3:instruct` via Ollama | Answer generation from text and table context |
| Vision Model | `llava` via Ollama | Chart/figure description at ingest + reading at query time |
| Embeddings | `nomic-embed-text` via Ollama | 768-dim vectors for semantic search |
| Vector DB | ChromaDB (persistent) | Store and query embeddings with cosine similarity |
| PDF Text | pymupdf (fitz) | Fast text extraction with positional sorting |
| PDF Tables | pdfplumber | Structural table detection → markdown conversion |
| PDF Images | pymupdf `get_images()` | Detect embedded raster images (charts/diagrams) |
| Figure Detection | Text-density analysis | Detect pages with charts by measuring text coverage |
| Chunking | langchain `RecursiveCharacterTextSplitter` | Split text at 800 chars with 150 overlap |
| Keyword Search | Custom BM25 implementation | Term-frequency scoring for exact-match queries |
| Rank Fusion | Reciprocal Rank Fusion (RRF) | Merge vector + BM25 results by rank |
| Reranker | LLM-as-judge (llama3) | Score chunk relevance 0-10, keep top 8 |
| API | FastAPI + uvicorn | REST API with auto-generated docs |
| Containerization | Docker Compose | Ollama + rag-app, GPU passthrough |

---

## Project Structure

```
DocumentRetriever/
├── app.py                      # FastAPI endpoints (POST /query, GET /sources)
├── ingest.py                   # Ingestion entry point
├── manage.py                   # CLI: setup, ingest, start, stop, status, logs
├── requirements.txt
├── data/
│   ├── pdfs/                   # Drop your SEC 10-Q PDFs here
│   └── chroma/                 # ChromaDB persistent storage + figure images
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── src/
│   ├── config.py               # All configuration (env vars)
│   ├── logger.py
│   ├── pipeline.py             # Orchestrates retrieve → rerank → generate
│   ├── ingestion/
│   │   ├── pdf_loader.py       # Multimodal PDF extraction (text/tables/figures)
│   │   └── vector_store.py     # Chunking, embedding, ChromaDB storage
│   ├── retrieval/
│   │   ├── retriever.py        # Hybrid search (vector + BM25 + RRF)
│   │   ├── bm25.py            # BM25 index + Reciprocal Rank Fusion
│   │   └── reranker.py        # LLM-based reranking
│   └── generation/
│       └── llm.py             # Dual-path generation (llama3 + llava)
├── static/
│   └── index.html             # Web UI
├── evaluation/
│   ├── evaluate.py            # Automated evaluation (16 metrics)
│   └── metrics.py
├── scripts/
│   ├── setup.sh
│   ├── ingest.sh
│   └── start.sh
├── RUNBOOK.md
└── .env.example
```

---

## How Each Modality is Handled

### Text
- **Detection**: Every page — `pymupdf.get_text("text", sort=True)`
- **Chunking**: Split at 800 chars with 150 char overlap
- **Embedding**: Full text chunk → `nomic-embed-text`
- **Generation**: All text chunks combined as context → `llama3:instruct`

### Tables
- **Detection**: `pdfplumber.find_tables()` — detects structured row/column layouts
- **Extraction**: Cell data → markdown format with headers and separators
- **Chunking**: Kept whole (not split) — tables lose meaning when fragmented
- **Embedding**: Markdown text → `nomic-embed-text`
- **Generation**: Markdown passed as context → `llama3:instruct` reads column/row structure

### Figures / Charts
- **Detection**: Two complementary methods:
  1. **Raster images**: `pymupdf.get_images()` — finds embedded PNG/JPEG charts (>200x200px)
  2. **Text-density**: If text covers <25% of a page, a large non-text element exists (vector chart, diagram)
- **Description**: Page rendered at 150 DPI → PNG → sent to `llava` → detailed text description stored
- **Storage**: Description text embedded for search; raw PNG saved to disk for query-time re-reading
- **Generation**: At query time, raw PNG image is sent back to `llava` with the user's specific question

---

## Configuration

All settings via environment variables (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://ollama:11434` | Ollama API endpoint |
| `LLM_MODEL` | `llama3:instruct` | Text/table generation model |
| `VISION_MODEL` | `llava` | Figure/chart vision model |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `CHUNK_SIZE` | `800` | Text chunk size (chars) |
| `CHUNK_OVERLAP` | `150` | Overlap between text chunks |
| `TOP_K` | `8` | Final chunks sent to LLM |
| `RERANK_ENABLED` | `true` | Enable LLM reranking |
| `RERANK_CANDIDATES` | `20` | Over-retrieve this many before reranking |

---

## API

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/query` | Ask a question → answer with source citations |
| `GET` | `/sources` | List all indexed PDF filenames |
| `GET` | `/health` | Health check |
| `GET` | `/` | Web UI |
| `GET` | `/docs` | Auto-generated API documentation |

### Query Request

```json
{
  "question": "What were total net sales in Q3 2022?",
  "top_k": 8,
  "source_filter": ["AAPL_10Q_2022Q3.pdf"]
}
```

### Query Response

```json
{
  "answer": "Total net sales were $83.0 billion...",
  "sources": ["AAPL_10Q_2022Q3.pdf"],
  "chunks": [
    {
      "source": "AAPL_10Q_2022Q3.pdf",
      "page": 4,
      "chunk_type": "table",
      "score": 0.89,
      "text_preview": "| Products | $65,085 | $63,722 |..."
    }
  ]
}
```

---

## Quick Start

```bash
cd DocumentRetriever

# 1. Add PDFs
cp /path/to/your/*.pdf data/pdfs/

# 2. One-time setup (~15 min, downloads models)
python manage.py setup

# 3. Ingest
python manage.py ingest

# 4. Query
open http://localhost:8000
```

See `RUNBOOK.md` for detailed commands, rebuild instructions, and troubleshooting.
