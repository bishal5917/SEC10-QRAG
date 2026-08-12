#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
COMPOSE_FILE="$PROJECT_DIR/docker/docker-compose.yml"
LOG_FILE="$PROJECT_DIR/setup.log"

# Tee all output to setup.log so it's never lost
exec > >(tee -a "$LOG_FILE") 2>&1
echo "======================================"
echo " Setup started: $(date)"
echo "======================================"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }

# ── 1. Check Docker ──────────────────────────────────────────────────────────
info "Checking Docker..."
if ! command -v docker &>/dev/null; then
    error "Docker not found. Install Docker Desktop from https://www.docker.com/products/docker-desktop/"
fi
docker info &>/dev/null || error "Docker daemon is not running. Please start Docker."
info "Docker OK"

# ── 2. Check NVIDIA Container Toolkit ────────────────────────────────────────
info "Checking NVIDIA GPU support..."
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -3 | while read gpu; do
        info "  GPU found: $gpu"
    done
    # Check nvidia-container-toolkit
    if ! docker run --rm --gpus all nvidia/cuda:12.0-base-ubuntu22.04 nvidia-smi &>/dev/null 2>&1; then
        warn "NVIDIA Container Toolkit may not be installed."
        warn "Install it from: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
        warn "Continuing anyway — Ollama will fall back to CPU if GPU is unavailable."
    else
        info "NVIDIA Container Toolkit OK"
    fi
else
    warn "nvidia-smi not found — will run on CPU (slower)."
fi

# ── 3. Build the RAG app image ────────────────────────────────────────────────
info "Cleaning up any stale containers..."
docker compose -f "$COMPOSE_FILE" down --remove-orphans 2>/dev/null || true
docker rm -f ollama rag-app 2>/dev/null || true
info "Build RAG app Docker image..."
docker compose -f "$COMPOSE_FILE" build --no-cache
info "Build complete"

# ── 4. Start Ollama service only first ───────────────────────────────────────
info "Starting Ollama service..."
docker compose -f "$COMPOSE_FILE" up -d ollama

info "Waiting for Ollama to be ready..."
for i in $(seq 1 30); do
    if curl -sf http://localhost:11434/api/tags &>/dev/null; then
        info "Ollama is ready"
        break
    fi
    if [ "$i" -eq 30 ]; then
        error "Ollama did not become ready in time. Check: docker logs ollama"
    fi
    sleep 3
done

# ── 5. Pull required Ollama models ───────────────────────────────────────────
info "Pulling LLM model: llama3:instruct (this may take a few minutes)..."
docker exec ollama ollama pull llama3:instruct

info "Pulling vision model: llava (this may take a few minutes)..."
docker exec ollama ollama pull llava

info "Pulling embedding model: nomic-embed-text..."
docker exec ollama ollama pull nomic-embed-text

info "Models ready:"
docker exec ollama ollama list

# ── 6. Start the full stack ───────────────────────────────────────────────────
info "Starting full stack (RAG app + Ollama)..."
docker compose -f "$COMPOSE_FILE" up -d

info ""
info "════════════════════════════════════════════"
info " Setup complete!"
info " RAG API:    http://localhost:8000"
info " API Docs:   http://localhost:8000/docs"
info " Ollama:     http://localhost:11434"
info ""
info " Next step: run ./scripts/ingest.sh to index your PDFs"
info "════════════════════════════════════════════"
