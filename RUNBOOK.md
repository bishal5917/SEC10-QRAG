# Runbook — Run & Evaluate

Step-by-step from a fresh machine to a full evaluation report.

---

## Only RUN one container 

- docker compose -f docker/docker-compose.yml build rag-app
- docker compose -f docker/docker-compose.yml up -d --remove-orphans rag-app

## Prerequisites

- Docker Desktop installed and running
- NVIDIA GPU + NVIDIA Container Toolkit *(optional — falls back to CPU)*
- Your PDFs ready
- `qna_data.csv` ground-truth file ready

---

## Step 1 — Open the project

```bash
cd /path/to/DocumentRetriever
chmod +x scripts/*.sh
```

---

## Step 2 — One-time setup

```bash
./scripts/setup.sh
```

What it does:
- Checks Docker and NVIDIA GPU
- Builds the Docker image (`unstructured`, `pymupdf`, `chromadb`, etc.)
- Starts Ollama
- Pulls `llama3:instruct` (~4.7GB), `llava` (~4.7GB), `nomic-embed-text` (~270MB)
- Starts the full stack in the background

> Takes ~15–20 min on first run due to model downloads.

---

## Step 3 — Add your PDFs

```bash
cp /path/to/your/*.pdf data/pdfs/
```

---

## Step 4 — Ingest PDFs

```bash
# PDFs already in data/pdfs/
./scripts/ingest.sh

# Or point to a directory
./scripts/ingest.sh /path/to/your/pdfs
```

What happens during ingest:
- `unstructured` partitions each PDF into Text / Table / Image / FigureCaption elements
- Geometry fallback detects vector charts (bar/line charts) that unstructured misses
- Tables rendered as markdown, figures rendered as PNG (base64)
- All chunks embedded via `nomic-embed-text` and stored in ChromaDB
- image_b64 stored in metadata for figure chunks

Logs will show chunk breakdown per PDF:
```
Loaded AAPL.pdf: 42 chunks {'text': 28, 'table': 13, 'figure': 1} in 3241ms
```

---

## Step 5 — Verify the stack is up

```bash
curl http://localhost:8000/health
# Expected: {"status":"ok"}

curl http://localhost:8000/sources
# Lists all indexed PDF filenames
```

---

## Step 6 — Test queries manually

```bash
# Text/table question → answered by llama3:instruct
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How has Apple total net sales changed over time?"}'

# Figure/chart question → answered by llava
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What does the INTC revenue trend chart show?"}'

# Filter to specific documents
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What were iPhone revenues?",
    "source_filter": ["2022 Q3 AAPL.pdf", "2023 Q1 AAPL.pdf"]
  }'
```

Or open the web UI: [http://localhost:8000](http://localhost:8000)

---

## Step 7 — Run evaluation

The evaluation script queries the live API and computes 16 metrics.

```bash
# Full evaluation (all answerable questions)
python evaluation/evaluate.py --csv /path/to/qna_data.csv

# Quick smoke-test (first 10 questions only)
python evaluation/evaluate.py --csv /path/to/qna_data.csv --limit 10

# Only table questions
python evaluation/evaluate.py --csv /path/to/qna_data.csv --chunk-type Table

# Only text questions
python evaluation/evaluate.py --csv /path/to/qna_data.csv --chunk-type Text

# Save full results to JSON
python evaluation/evaluate.py --csv /path/to/qna_data.csv --output results.json
```

Metrics reported:

| Group | Metrics |
|---|---|
| Retrieval | Context Precision, Recall, F1, MRR, nDCG |
| Generation | BLEU, ROUGE-1/2/L, METEOR, BERTScore |
| End-to-end | Faithfulness, Hallucination Rate, Factual Consistency, Answer Relevance, Exact Number Match |

> Note: The evaluation CSV contains only Text and Table questions. Figure/chart questions are not in the CSV — llava is exercised only when chart pages are retrieved for relevant queries.

---

## Day-to-day commands

```bash
# Start stack with live logs (day-to-day)
./scripts/start.sh          # Ctrl+C to stop

# Start stack in background
docker compose -f docker/docker-compose.yml up -d

# Stop everything
docker compose -f docker/docker-compose.yml down

# Re-ingest after adding new PDFs
./scripts/ingest.sh

# View logs
docker logs rag-app -f
docker logs ollama -f

# List indexed sources
curl http://localhost:8000/sources

# Check which Ollama models are loaded
docker exec ollama ollama list
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Cannot reach API at http://localhost:8000` | Run `./scripts/start.sh` or `docker compose -f docker/docker-compose.yml up -d` |
| `No PDFs found in data/pdfs` | Copy PDFs first — Step 3 |
| Ollama model not found | `docker exec ollama ollama pull llama3:instruct` |
| llava not found | `docker exec ollama ollama pull llava` |
| GPU not detected | Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) — falls back to CPU otherwise |
| BERTScore slow on first eval run | Downloads `roberta-large` once — subsequent runs are fast |
| Ingest slow | Normal — `unstructured` partitions each page, figure pages also render PNG |
| Figure chunk not detected | Page may have a greyscale or scanned chart — switch to `strategy="hi_res"` in `pdf_loader.py` for better detection |
