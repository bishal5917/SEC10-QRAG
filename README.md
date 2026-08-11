# Document Retriever — Multimodal RAG System

RAG-based information retrieval over PDF documents. Answers questions about text, tables, and figures/charts using a hybrid multimodal pipeline — `llama3:instruct` for text and tables, `llava` for charts and diagrams.

## Architecture

```
PDF files
   │
   ▼
pdf_loader.py   ← unstructured detects element types (Text/Table/Image/FigureCaption)
                   + geometry fallback for vector charts (bar/line charts)
                   tables → markdown via BeautifulSoup
                   figures → PNG rendered at 150 DPI → base64
   │
   ▼
vector_store.py ← text chunks split (800 chars), table/figure kept whole
                   all embedded via nomic-embed-text → stored in ChromaDB
                   image_b64 stored in metadata (not embedded)
   │
   ▼
retriever.py    ← cosine similarity search, optional source filter
                   image_b64 passed through from metadata
   │
   ▼
llm.py          ← routes by chunk type:
                   text/table → llama3:instruct
                   figure     → llava (receives PNG image)
   │
   ▼
app.py          ← FastAPI: POST /query
```

## Models

| Model | Purpose |
|---|---|
| `llama3:instruct` | Text and table question answering |
| `llava` | Chart and figure visual understanding |
| `nomic-embed-text` | Embeddings for all chunk types |

## Chunk Types

| Type | Detection | Storage | Generation |
|---|---|---|---|
| `text` | unstructured NarrativeText/Title/ListItem | text string, split at 800 chars | llama3:instruct |
| `table` | unstructured Table element | markdown string, not split | llama3:instruct |
| `figure` | unstructured Image/FigureCaption + geometry fallback for vector charts | spatial text (for retrieval) + PNG base64 in metadata | llava |

## Prerequisites

- Docker Desktop (with Compose v2)
- NVIDIA GPU + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) *(optional but recommended)*

## Quick Start

### 1. Setup (one-time)

```bash
chmod +x scripts/*.sh
./scripts/setup.sh
```

This will:
- Build the Docker image
- Start Ollama
- Pull `llama3:instruct`, `llava`, and `nomic-embed-text`
- Start the full stack

> Takes ~10–15 min on first run — `llava` is ~4.7GB.

### 2. Add your PDFs

```bash
cp /path/to/your/*.pdf data/pdfs/
```

### 3. Ingest PDFs

```bash
./scripts/ingest.sh

# Or point directly to a directory
./scripts/ingest.sh /path/to/your/pdfs
```

### 4. Query the API

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How has Apple total net sales changed over time?"}'
```

Filter to specific documents:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What were the iPhone revenues?",
    "source_filter": ["2022 Q3 AAPL.pdf", "2023 Q1 AAPL.pdf"]
  }'
```

### 5. Interactive API Docs

Open [http://localhost:8000/docs](http://localhost:8000/docs)

## Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/query` | Ask a question, get an answer with citations |
| GET | `/sources` | List all indexed PDF filenames |
| GET | `/health` | Health check |

## Configuration

All settings in `src/config.py`, overridable via environment variables (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `LLM_MODEL` | `llama3:instruct` | Ollama model for text/table generation |
| `VISION_MODEL` | `llava` | Ollama model for figure/chart generation |
| `EMBED_MODEL` | `nomic-embed-text` | Ollama model for embeddings |
| `CHUNK_SIZE` | `800` | Text chunk size in characters |
| `CHUNK_OVERLAP` | `150` | Overlap between text chunks |
| `TOP_K` | `8` | Number of chunks retrieved per query |

## Evaluation

```bash
# Full evaluation
python evaluation/evaluate.py --csv /path/to/qna_data.csv

# Quick smoke-test
python evaluation/evaluate.py --csv /path/to/qna_data.csv --limit 10

# Save results
python evaluation/evaluate.py --csv /path/to/qna_data.csv --output results.json
```

16 metrics across retrieval, generation, and end-to-end categories. See `RUNBOOK.md` for full evaluation guide.

## Useful Commands

```bash
# Stream live logs
./scripts/start.sh

# Stop everything
docker compose -f docker/docker-compose.yml down

# Re-ingest after adding new PDFs
./scripts/ingest.sh

# View logs
docker logs rag-app -f
docker logs ollama -f

# List indexed sources
curl http://localhost:8000/sources
```
