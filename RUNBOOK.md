# Runbook

## Prerequisites

- Docker Desktop installed and running
- NVIDIA GPU + Container Toolkit *(optional — falls back to CPU)*

---

## Quick Start (from scratch)

```bash
cd /path/to/DocumentRetriever

# 1. Add your PDFs
cp /path/to/your/*.pdf data/pdfs/

# 2. One-time setup (builds image, pulls models ~15 min first time)
python manage.py setup

# 3. Ingest PDFs into vector store
python manage.py ingest

# 4. Query via API or web UI
open http://localhost:8000
```

---

## What Happens During Ingest

Each PDF page is analyzed for:
- **Text** — extracted via pymupdf
- **Tables** — detected by pdfplumber, converted to markdown
- **Figures/Charts** — detected by text-density analysis + embedded image detection, rendered at 150 DPI, described by llava

All chunks are embedded with `nomic-embed-text` and stored in ChromaDB.

---

## Day-to-Day Commands

```bash
python manage.py start     # Start stack (live logs, Ctrl+C to stop)
python manage.py stop      # Stop everything
python manage.py status    # Check container states
python manage.py logs      # Tail logs
python manage.py ingest    # Re-ingest after adding new PDFs
```

---

## Rebuild After Code Changes (without full setup)

If you already have Ollama + models set up and only changed Python code:

```bash
cd /path/to/DocumentRetriever

# Rebuild just the app image (keeps ollama + models intact)
docker compose -f docker/docker-compose.yml build rag-app

# Clean up dangling <none> images
docker image prune -f

# Restart with new code
docker compose -f docker/docker-compose.yml up -d

# Re-ingest (new extraction logic needs fresh chunks)
docker compose -f docker/docker-compose.yml exec rag-app python ingest.py
```

> No need to re-pull Ollama models. Only takes ~30 seconds for the rebuild.

---

## Query Examples

```bash
# Text/table question
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What were total net sales in Q3 2022?"}'

# Filter to specific documents
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What were iPhone revenues?", "source_filter": ["AAPL_10Q.pdf"]}'
```

---

## Retrieval Pipeline

```
Query → Hybrid Search (Vector + BM25 keyword) → RRF Fusion → LLM Reranker → Top 8 chunks
  ├─ Text/Table chunks → llama3:instruct → answer
  └─ Figure chunks → llava (with raw image) → answer
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| API unreachable | `python manage.py start` |
| No PDFs found | Copy PDFs to `data/pdfs/` first |
| Model not found | `docker exec ollama ollama pull llama3:instruct` |
| GPU not detected | Install NVIDIA Container Toolkit — CPU fallback works |
| Ingest slow | Normal — figure pages call llava for descriptions |
| Check logs | `python manage.py logs` or see `manage.log` in project root |
