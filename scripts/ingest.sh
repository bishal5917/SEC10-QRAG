#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
COMPOSE_FILE="$PROJECT_DIR/docker/docker-compose.yml"
PDF_DIR="$PROJECT_DIR/data/pdfs"
LOG_FILE="$PROJECT_DIR/ingest.log"

exec > >(tee -a "$LOG_FILE") 2>&1
echo "======================================"
echo " Ingest started: $(date)"
echo "======================================"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }


# ── Check PDFs exist ──────────────────────────────────────────────────────────
PDF_COUNT=$(find "$PDF_DIR" -name "*.pdf" 2>/dev/null | wc -l | tr -d ' ')
if [ "$PDF_COUNT" -eq 0 ]; then
    error "No PDFs found in $PDF_DIR"
fi
info "Found $PDF_COUNT PDF(s) in $PDF_DIR"

# ── Check rag-app container is running ────────────────────────────────────────
if ! docker compose -f "$COMPOSE_FILE" ps rag-app | grep -q "running"; then
    warn "rag-app container is not running. Starting it..."
    docker compose -f "$COMPOSE_FILE" up -d
    sleep 5
fi

# ── Run ingestion inside the container ───────────────────────────────────────
info "Running ingestion pipeline..."
docker compose -f "$COMPOSE_FILE" exec rag-app python ingest.py

info ""
info "════════════════════════════════════════════"
info " Ingestion complete!"
info " You can now query the API:"
info ""
info ' curl -X POST http://localhost:8000/query \'
info '   -H "Content-Type: application/json" \'
info '   -d '"'"'{"question": "How has Apple total net sales changed over time?"}'"'"
info "════════════════════════════════════════════"
