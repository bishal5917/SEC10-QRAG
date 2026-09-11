"""
Ingestion CLI Script.

Processes all PDFs in data/pdfs/, extracts text/tables/images,
computes embeddings, and stores everything in Qdrant (embedded, hybrid).

Run this once (or whenever PDFs change) before querying:
    python ingest.py            # Ingest (append to existing)
    python ingest.py --clear    # Clear store and re-ingest from scratch

Pipeline Steps:
    1. Load text from PDFs (PyMuPDF) → split into chunks (LangChain splitter)
    2. Extract tables (pdfplumber) → convert to Markdown
    3. Render page images (PyMuPDF) → embed with CLIP
    4. Embed text/tables with BGE (dense) + BM25 (sparse) → store in Qdrant
    5. Store CLIP image embeddings in a separate Qdrant collection
"""

import argparse
import sys
import time

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from app.core.config import settings
from app.core.logging import get_logger, setup_logging
from app.document_processing.image_extractor import PageImageExtractor
from app.document_processing.pdf_loader import PDFDirectoryLoader
from app.document_processing.table_extractor import TableExtractor
from app.document_processing.text_splitter import split_documents
from app.embeddings.clip_embeddings import CLIPImageEmbeddings
from app.embeddings.text_embeddings import BGETextEmbeddings
from app.vector_store.store import MultiModalVectorStore

console = Console()
logger = get_logger(__name__)


def run_ingestion(clear_existing: bool = False) -> None:
    """
    Execute the full ingestion pipeline.

    Args:
        clear_existing: If True, deletes all existing vector store data
                        before starting ingestion.
    """
    start_time = time.time()

    console.print(Panel.fit(
        "[bold]🔄 Multimodal RAG — Ingestion Pipeline[/bold]\n"
        "PDF → Text/Tables/Images → Embeddings → Qdrant (hybrid)",
        border_style="blue",
    ))

    # ─── Step 1: Initialize models ───────────────────────────────────────
    console.print("\n[bold]Step 1:[/bold] Loading embedding models...")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading BGE text embeddings...", total=None)
        text_embeddings = BGETextEmbeddings()

        progress.update(task, description="Loading CLIP image embeddings...")
        clip_embeddings = CLIPImageEmbeddings()

        progress.update(task, description="Connecting to Qdrant (embedded)...")
        vector_store = MultiModalVectorStore(
            text_embeddings=text_embeddings,
            image_embeddings=clip_embeddings,
        )

    if clear_existing:
        console.print("[yellow]⚠ Clearing existing vector store...[/yellow]")
        # clear_all() drops and recreates collections on the SAME client
        # (Qdrant embedded mode allows only one client per folder).
        vector_store.clear_all()

    # ─── Step 2: Extract text and split into chunks ──────────────────────
    console.print("\n[bold]Step 2:[/bold] Extracting and chunking text from PDFs...")
    pdf_loader = PDFDirectoryLoader()
    raw_documents = pdf_loader.load()
    text_chunks = split_documents(raw_documents)
    console.print(f"  ✓ {len(text_chunks)} text chunks created")

    # ─── Step 3: Extract tables ──────────────────────────────────────────
    console.print("\n[bold]Step 3:[/bold] Extracting tables from PDFs...")
    table_extractor = TableExtractor()
    table_documents = table_extractor.extract_from_directory()
    console.print(f"  ✓ {len(table_documents)} tables extracted")

    # ─── Step 4: Render page images ─────────────────────────────────────
    console.print("\n[bold]Step 4:[/bold] Rendering page images...")
    image_extractor = PageImageExtractor()
    image_documents = image_extractor.extract_from_directory()
    console.print(f"  ✓ {len(image_documents)} page images rendered")

    # ─── Step 5: Embed and store text chunks ─────────────────────────────
    console.print("\n[bold]Step 5:[/bold] Embedding and storing text chunks...")
    vector_store.add_text_documents(text_chunks)
    console.print(f"  ✓ Text chunks stored in Qdrant")

    # ─── Step 6: Embed and store tables ──────────────────────────────────
    console.print("\n[bold]Step 6:[/bold] Embedding and storing tables...")
    vector_store.add_table_documents(table_documents)
    console.print(f"  ✓ Tables stored in Qdrant")

    # ─── Step 7: Embed and store page images (CLIP) ─────────────────────
    console.print("\n[bold]Step 7:[/bold] Embedding page images with CLIP...")
    image_paths = [doc.metadata["image_path"] for doc in image_documents]
    image_vectors = clip_embeddings.embed_images(image_paths)
    vector_store.add_image_documents(image_documents, image_vectors)
    console.print(f"  ✓ Image embeddings stored in Qdrant")

    # ─── Summary ─────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    stats = vector_store.get_stats()

    console.print(Panel.fit(
        f"[bold green]✓ Ingestion Complete![/bold green]\n\n"
        f"Text chunks: {stats['text']}\n"
        f"Tables:      {stats['tables']}\n"
        f"Images:      {stats['images']}\n"
        f"Time:        {elapsed:.1f}s\n\n"
        f"Run [bold]python query.py[/bold] to start asking questions.",
        border_style="green",
    ))

    # Close the embedded Qdrant client cleanly to avoid shutdown-time noise
    vector_store.close()


def main():
    """CLI entry point for ingestion."""
    parser = argparse.ArgumentParser(
        description="Ingest PDFs into the multimodal RAG vector store"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing vector store before ingestion",
    )
    args = parser.parse_args()

    # Configure logging from config (settings.log_level)
    setup_logging()

    # Validate configuration
    if not settings.pdf_dir.exists():
        console.print(
            f"[bold red]Error:[/bold red] PDF directory not found: {settings.pdf_dir}"
        )
        sys.exit(1)

    pdf_count = len(list(settings.pdf_dir.glob("*.pdf")))
    if pdf_count == 0:
        console.print(
            f"[bold red]Error:[/bold red] No PDF files found in {settings.pdf_dir}"
        )
        sys.exit(1)

    console.print(f"[dim]Found {pdf_count} PDFs in {settings.pdf_dir}[/dim]")

    try:
        run_ingestion(clear_existing=args.clear)
    except Exception as e:
        console.print(f"\n[bold red]Fatal Error:[/bold red] {e}")
        logger.exception("Ingestion failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
