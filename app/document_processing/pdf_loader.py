"""
PDF Document Loader Module.

Provides a LangChain-compatible document loader that extracts text content
from PDF files using PyMuPDF. Each page becomes a separate LangChain Document
with rich metadata (source file, page number, company ticker, quarter, year).

Architecture:
    PDFLoader wraps PyMuPDF (fitz) and produces LangChain Document objects,
    making it seamlessly compatible with LangChain's text splitters,
    chains, and retrieval pipelines.
"""

from pathlib import Path
from typing import Iterator

import pymupdf  # PyMuPDF
from langchain_core.documents import Document

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def _extract_metadata_from_filename(filename: str) -> dict:
    """
    Parse structured metadata from the PDF filename.

    Expected filename format: '{YEAR} {QUARTER} {TICKER}.pdf'
    Example: '2023 Q3 AAPL.pdf' → {'year': '2023', 'quarter': 'Q3', 'ticker': 'AAPL'}

    Args:
        filename: Name of the PDF file (with extension).

    Returns:
        Dictionary with extracted metadata fields.
        Falls back to just filename if pattern doesn't match.
    """
    stem = Path(filename).stem
    parts = stem.split()

    metadata = {"filename": filename}

    if len(parts) >= 3:
        metadata["year"] = parts[0]
        metadata["quarter"] = parts[1]
        metadata["ticker"] = parts[2]

    return metadata


class PDFTextLoader:
    """
    LangChain-compatible document loader for PDF text extraction.

    Uses PyMuPDF for fast, accurate text extraction. Produces one Document
    per page, preserving page boundaries for downstream chunking.

    Usage:
        loader = PDFTextLoader(Path("data/pdfs/2023 Q3 AAPL.pdf"))
        documents = loader.load()  # List[Document]

    Each document has metadata:
        - source: file path
        - page_number: 1-indexed page number
        - total_pages: total pages in the PDF
        - year, quarter, ticker: parsed from filename
    """

    def __init__(self, file_path: Path):
        """
        Initialize the PDF text loader.

        Args:
            file_path: Path to the PDF file to load.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"PDF not found: {self.file_path}")

        self.metadata = _extract_metadata_from_filename(self.file_path.name)

    def load(self) -> list[Document]:
        """
        Load and extract text from all pages of the PDF.

        Returns:
            List of Document objects, one per page with non-empty text.
        """
        documents = []

        try:
            doc = pymupdf.open(self.file_path)
            total_pages = len(doc)

            for page_idx in range(total_pages):
                page = doc[page_idx]
                text = page.get_text("text").strip()

                # Skip pages with no meaningful text content
                if not text or len(text) < 20:
                    continue

                page_metadata = {
                    "source": str(self.file_path),
                    "page_number": page_idx + 1,
                    "total_pages": total_pages,
                    **self.metadata,
                }

                documents.append(Document(page_content=text, metadata=page_metadata))

            doc.close()
            logger.debug(
                f"Loaded {len(documents)} pages from {self.file_path.name}"
            )

        except Exception as e:
            logger.error(f"Failed to load {self.file_path.name}: {e}")

        return documents

    def lazy_load(self) -> Iterator[Document]:
        """
        Lazily load pages one at a time (memory-efficient for large PDFs).

        Yields:
            Document objects one page at a time.
        """
        try:
            doc = pymupdf.open(self.file_path)
            total_pages = len(doc)

            for page_idx in range(total_pages):
                page = doc[page_idx]
                text = page.get_text("text").strip()

                if not text or len(text) < 20:
                    continue

                page_metadata = {
                    "source": str(self.file_path),
                    "page_number": page_idx + 1,
                    "total_pages": total_pages,
                    **self.metadata,
                }

                yield Document(page_content=text, metadata=page_metadata)

            doc.close()

        except Exception as e:
            logger.error(f"Failed to lazy-load {self.file_path.name}: {e}")


class PDFDirectoryLoader:
    """
    Loads all PDFs from a directory, producing Documents for each page.

    Wraps PDFTextLoader for batch processing of an entire PDF directory.

    Usage:
        loader = PDFDirectoryLoader(settings.pdf_dir)
        all_documents = loader.load()
    """

    def __init__(self, directory: Path = None):
        """
        Initialize the directory loader.

        Args:
            directory: Path to directory containing PDFs.
                       Defaults to settings.pdf_dir.
        """
        self.directory = directory or settings.pdf_dir

        if not self.directory.exists():
            raise FileNotFoundError(f"PDF directory not found: {self.directory}")

    def get_pdf_files(self) -> list[Path]:
        """Get sorted list of all PDF files in the directory."""
        pdfs = sorted(self.directory.glob("*.pdf"))
        logger.info(f"Found {len(pdfs)} PDF files in {self.directory}")
        return pdfs

    def load(self) -> list[Document]:
        """
        Load text from all PDFs in the directory.

        Returns:
            Combined list of Documents from all PDFs.
        """
        all_documents = []
        pdf_files = self.get_pdf_files()

        for pdf_path in pdf_files:
            loader = PDFTextLoader(pdf_path)
            documents = loader.load()
            all_documents.extend(documents)

        logger.info(
            f"Total text documents loaded: {len(all_documents)} "
            f"from {len(pdf_files)} PDFs"
        )
        return all_documents
