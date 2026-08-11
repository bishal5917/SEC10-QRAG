"""
Run this once to ingest all PDFs into the vector store.
Usage: python ingest.py
"""
import time
from src.config import DATA_DIR
from src.ingestion.pdf_loader import load_all_pdfs
from src.ingestion.vector_store import build_vector_store
from src.logger import get_logger

log = get_logger("ingest")


def main():
    log.info("=" * 60)
    log.info("Starting ingestion pipeline")
    log.info(f"PDF source directory: {DATA_DIR}")
    log.info("=" * 60)

    t0 = time.perf_counter()

    log.info("[1/2] Loading and extracting PDFs...")
    chunks = load_all_pdfs(DATA_DIR)
    log.info(f"[1/2] Extraction complete: {len(chunks)} raw chunks")

    log.info("[2/2] Building vector store...")
    build_vector_store(chunks)

    log.info("=" * 60)
    log.info(f"Ingestion complete in {time.perf_counter() - t0:.1f}s")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
