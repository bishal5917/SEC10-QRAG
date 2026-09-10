"""
Page Image Extraction Module.

Renders each PDF page as a high-resolution PNG image using PyMuPDF.
These page images are embedded with CLIP for visual retrieval and can
be sent directly to Gemini for multimodal reasoning (charts, graphs, etc.).

Architecture:
    PDF Page → PyMuPDF render (200 DPI) → PNG file → CLIP embedding
                                                   → Gemini visual input
"""

from pathlib import Path

import pymupdf  # PyMuPDF
from langchain_core.documents import Document

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def _extract_metadata_from_filename(filename: str) -> dict:
    """Parse year/quarter/ticker from filename pattern '{YEAR} {QUARTER} {TICKER}.pdf'."""
    stem = Path(filename).stem
    parts = stem.split()

    metadata = {"filename": filename}
    if len(parts) >= 3:
        metadata["year"] = parts[0]
        metadata["quarter"] = parts[1]
        metadata["ticker"] = parts[2]

    return metadata


class PageImageExtractor:
    """
    Extracts page images from PDFs by rendering each page as PNG.

    The rendered images serve two purposes in the pipeline:
        1. Embedded with CLIP for visual similarity retrieval.
        2. Sent as visual context to Gemini for chart/graph interpretation.

    Usage:
        extractor = PageImageExtractor()
        image_docs = extractor.extract_from_pdf(Path("2023 Q3 AAPL.pdf"))
        # image_docs[0].metadata["image_path"] → path to the PNG file
    """

    def __init__(self, output_dir: Path = None):
        """
        Initialize the image extractor.

        Args:
            output_dir: Directory to save rendered page images.
                        Defaults to settings.images_dir.
        """
        self.output_dir = output_dir or settings.images_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_from_pdf(self, pdf_path: Path) -> list[Document]:
        """
        Render each page of a PDF as a PNG image.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            List of Documents with metadata containing the image_path.
            The page_content field contains a text description for reference.
        """
        documents = []
        file_metadata = _extract_metadata_from_filename(pdf_path.name)
        pdf_stem = pdf_path.stem.replace(" ", "_")

        try:
            doc = pymupdf.open(pdf_path)
            # Calculate zoom factor: target DPI / default 72 DPI
            zoom = settings.page_image_dpi / 72.0
            matrix = pymupdf.Matrix(zoom, zoom)

            for page_idx in range(len(doc)):
                page = doc[page_idx]
                page_number = page_idx + 1

                # Render page to pixmap (in-memory image)
                pixmap = page.get_pixmap(matrix=matrix)

                # Save as PNG
                image_filename = f"{pdf_stem}_page_{page_number}.png"
                image_path = self.output_dir / image_filename
                pixmap.save(str(image_path))

                # Create a Document referencing the saved image
                # page_content is a textual description for logging/debugging
                page_content = (
                    f"Page image from {pdf_path.name}, page {page_number}. "
                    f"Contains visual content including potential charts, "
                    f"tables, and figures."
                )

                metadata = {
                    "source": str(pdf_path),
                    "image_path": str(image_path),
                    "page_number": page_number,
                    "total_pages": len(doc),
                    "content_type": "page_image",
                    **file_metadata,
                }

                documents.append(
                    Document(page_content=page_content, metadata=metadata)
                )

            doc.close()
            logger.debug(
                f"Rendered {len(documents)} page images from {pdf_path.name}"
            )

        except Exception as e:
            logger.error(f"Image extraction failed for {pdf_path.name}: {e}")

        return documents

    def extract_from_directory(self, directory: Path = None) -> list[Document]:
        """
        Extract page images from all PDFs in a directory.

        Args:
            directory: Path to directory with PDFs. Defaults to settings.pdf_dir.

        Returns:
            Combined list of image Documents from all PDFs.
        """
        directory = directory or settings.pdf_dir
        pdf_files = sorted(directory.glob("*.pdf"))
        all_images = []

        for pdf_path in pdf_files:
            images = self.extract_from_pdf(pdf_path)
            all_images.extend(images)

        logger.info(
            f"Total page images extracted: {len(all_images)} from {len(pdf_files)} PDFs"
        )
        return all_images
