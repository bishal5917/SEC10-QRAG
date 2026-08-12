"""
manage.py — Cross-platform management script for the RAG system.
Replaces setup.sh, ingest.sh, and start.sh.

Usage:
    python manage.py setup    # one-time: build image, pull models, start stack
    python manage.py ingest   # index PDFs into ChromaDB
    python manage.py start    # start stack with live logs
    python manage.py stop     # stop everything
    python manage.py status   # show container states
    python manage.py logs     # tail logs from rag-app and ollama
"""

import subprocess
import sys
import time
import urllib.request
import urllib.error
import logging
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_DIR  = Path(__file__).resolve().parent
COMPOSE_FILE = PROJECT_DIR / "docker" / "docker-compose.yml"
PDF_DIR      = PROJECT_DIR / "data" / "pdfs"
LOG_FILE     = PROJECT_DIR / "manage.log"

# ── Logging — prints to terminal AND saves to manage.log ──────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
log = logging.getLogger("manage")


def run(cmd: list, check: bool = True, stream: bool = False) -> subprocess.CompletedProcess:
    """Run a command, stream output live if requested, log errors clearly."""
    log.info(f"$ {' '.join(str(c) for c in cmd)}")
    if stream:
        # Stream output directly to terminal (for docker compose up)
        proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
        proc.wait()
        return proc
    result = subprocess.run(cmd, capture_output=False, text=True)
    if check and result.returncode != 0:
        log.error(f"Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    return result


def compose(*args) -> list:
    return ["docker", "compose", "-f", str(COMPOSE_FILE)] + list(args)


def wait_for_ollama(retries: int = 30, delay: int = 3) -> bool:
    log.info("Waiting for Ollama to be ready...")
    for i in range(retries):
        try:
            urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3)
            log.info("Ollama is ready")
            return True
        except Exception:
            log.info(f"  Not ready yet ({i+1}/{retries}), retrying in {delay}s...")
            time.sleep(delay)
    log.error("Ollama did not become ready in time. Check: python manage.py logs")
    return False


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_setup():
    log.info("=" * 55)
    log.info(" SETUP — build image, pull models, start stack")
    log.info("=" * 55)

    # 1. Check Docker
    log.info("Checking Docker...")
    result = subprocess.run(["docker", "info"], capture_output=True)
    if result.returncode != 0:
        log.error("Docker is not running. Please start Docker Desktop.")
        sys.exit(1)
    log.info("Docker OK")

    # 2. Check NVIDIA GPU
    result = subprocess.run(["nvidia-smi"], capture_output=True)
    if result.returncode == 0:
        log.info("NVIDIA GPU detected")
    else:
        log.warning("nvidia-smi not found — will run on CPU (slower)")

    # 3. Clean up stale containers
    log.info("Cleaning up stale containers...")
    run(compose("down", "--remove-orphans"), check=False)
    subprocess.run(["docker", "rm", "-f", "ollama", "rag-app"], capture_output=True)

    # 3b. Remove dangling images (<none> tags) to free disk space
    log.info("Removing dangling images...")
    subprocess.run(["docker", "image", "prune", "-f"], capture_output=True)

    # 4. Build image
    log.info("Building Docker image (this may take a few minutes)...")
    run(compose("build", "--no-cache"))
    log.info("Build complete")

    # 5. Start Ollama first
    log.info("Starting Ollama...")
    run(compose("up", "-d", "ollama"))

    if not wait_for_ollama():
        sys.exit(1)

    # 6. Pull models
    for model, label in [
        ("llama3:instruct", "LLM model"),
        ("llava",           "Vision model"),
        ("nomic-embed-text","Embedding model"),
    ]:
        log.info(f"Pulling {label}: {model} ...")
        run(["docker", "exec", "ollama", "ollama", "pull", model])

    log.info("Models ready:")
    run(["docker", "exec", "ollama", "ollama", "list"])

    # 7. Start full stack
    log.info("Starting full stack...")
    run(compose("up", "-d", "--remove-orphans"))

    log.info("")
    log.info("=" * 55)
    log.info(" Setup complete!")
    log.info("   RAG API  : http://localhost:8000")
    log.info("   API Docs : http://localhost:8000/docs")
    log.info("   Ollama   : http://localhost:11434")
    log.info("")
    log.info(" Next: python manage.py ingest")
    log.info("=" * 55)


def cmd_ingest(pdf_source: str = None):
    log.info("=" * 55)
    log.info(" INGEST — index PDFs into ChromaDB")
    log.info("=" * 55)

    # Optionally copy PDFs from another directory
    if pdf_source:
        src = Path(pdf_source)
        if not src.exists():
            log.error(f"Source directory not found: {src}")
            sys.exit(1)
        PDF_DIR.mkdir(parents=True, exist_ok=True)
        pdfs = list(src.glob("*.pdf"))
        if not pdfs:
            log.warning(f"No PDFs found in {src}")
        for pdf in pdfs:
            import shutil
            shutil.copy(pdf, PDF_DIR / pdf.name)
            log.info(f"  Copied: {pdf.name}")

    # Check PDFs exist
    pdfs = list(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        log.error(f"No PDFs found in {PDF_DIR}. Copy PDFs first or run:")
        log.error(f"  python manage.py ingest /path/to/your/pdfs")
        sys.exit(1)
    log.info(f"Found {len(pdfs)} PDF(s) in {PDF_DIR}")

    # Check rag-app is running
    result = subprocess.run(
        compose("ps", "rag-app"),
        capture_output=True, text=True
    )
    if "running" not in result.stdout.lower():
        log.warning("rag-app is not running — starting stack...")
        run(compose("up", "-d", "--remove-orphans"))
        time.sleep(5)

    # Run ingestion inside container
    log.info("Running ingestion pipeline...")
    run(compose("exec", "rag-app", "python", "ingest.py"))

    log.info("")
    log.info("=" * 55)
    log.info(" Ingestion complete!")
    log.info(' Test: curl -X POST http://localhost:8000/query \\')
    log.info('         -H "Content-Type: application/json" \\')
    log.info('         -d \'{"question": "What were the total net sales?"}\'')
    log.info("=" * 55)


def cmd_start():
    log.info("Starting stack — logs streaming live. Press Ctrl+C to stop.")
    result = subprocess.run(["docker", "image", "inspect", "rag-app"], capture_output=True)
    if result.returncode != 0:
        log.error("Image not found. Run: python manage.py setup")
        sys.exit(1)
    run(compose("up", "--remove-orphans", "--no-log-prefix"), stream=True)


def cmd_stop():
    log.info("Stopping stack...")
    run(compose("down"))
    log.info("Stack stopped.")


def cmd_status():
    log.info("Container status:")
    run(compose("ps"))


def cmd_logs():
    log.info("Tailing logs (Ctrl+C to stop)...")
    run(compose("logs", "-f", "--tail=50"), stream=True)


# ── Entry point ───────────────────────────────────────────────────────────────

COMMANDS = {
    "setup":  cmd_setup,
    "ingest": cmd_ingest,
    "start":  cmd_start,
    "stop":   cmd_stop,
    "status": cmd_status,
    "logs":   cmd_logs,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        sys.exit(0)

    command = sys.argv[1]
    extra   = sys.argv[2] if len(sys.argv) > 2 else None

    log.info(f"manage.py {command} — log saved to {LOG_FILE}")

    if command == "ingest" and extra:
        cmd_ingest(pdf_source=extra)
    else:
        COMMANDS[command]()
