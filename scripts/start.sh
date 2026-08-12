#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
COMPOSE_FILE="$PROJECT_DIR/docker/docker-compose.yml"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info() { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC}  $*"; }

# Make sure the stack is built first
if ! docker image inspect rag-app &>/dev/null; then
    warn "Image not found. Run ./scripts/setup.sh first."
    exit 1
fi

info "Starting stack — logs will stream live below."
info "Press Ctrl+C to stop everything."
info ""

# --no-log-prefix shows raw logs without the 'container_name |' prefix per line
# Remove it if you want to see which container each line comes from
exec docker compose -f "$COMPOSE_FILE" up --no-log-prefix
