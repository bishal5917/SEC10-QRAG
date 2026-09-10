"""
Table Extraction Module.

Extracts tables from PDF documents using pdfplumber and converts them
to Markdown format for embedding and LLM consumption. Tables are
stored as LangChain Documents with metadata linking them to their
source PDF and page.

Design Rationale:
    - Markdown tables are the best format for LLM consumption (readable, structured).
    - pdfplumber is used over camelot for better handling of complex table layouts.
    - Each table becomes a separate Document, enabling fine-grained table retrieval.
"""

from pathlib import Path
from typing import Optional

import pdfplumber
from langchain_core.documents import Document

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def _table_to_markdown(table: list[list], source_context: str = "") -> Optional[str]:
    """
    Convert a pdfplumber table (list of rows) to Markdown format.

    Args:
        table: Raw table data as a list of rows, where each row is a list of cell values.
        source_context: Optional context string prepended to the table.

    Returns:
        Markdown-formatted table string, or None if the table is too small/empty.
    """
    # Skip tables with fewer than 2 rows (need at least header + one data row)
    if not table or len(table) < 2:
        return None

    # Clean cells: replace None/empty with "-", strip whitespace
    cleaned_rows = []
    for row in table:
        cleaned_row = []
        for cell in row:
            if cell is None or str(cell).strip() == "":
                cleaned_row.append("-")
            else:
                # Collapse multi-line cells and strip whitespace
                cleaned_row.append(str(cell).replace("\n", " ").strip())
        cleaned_rows.append(cleaned_row)

    # Determine column count from the header (first row)
    num_cols = len(cleaned_rows[0])

    # Skip if all cells are empty/dashes
    non_empty_cells = sum(
        1 for row in cleaned_rows for cell in row if cell != "-"
    )
    if non_empty_cells < 3:
        return None

    # Build the Markdown table
    lines = []

    # Optional context header
    if source_context:
        lines.append(f"**{source_context}**\n")

    # Header row
    header = cleaned_rows[0][:num_cols]
    lines.append("| " + " | ".join(header) + " |")

    # Separator row
    lines.append("| " + " | ".join(["---"] * num_cols) + " |")

    # Data rows
    for row in cleaned_rows[1:]:
        # Pad or truncate row to match header column count
        padded_row = (row + ["-"] * num_cols)[:num_cols]
        lines.append("| " + " | ".join(padded_row) + " |")

    return "\n".join(lines)


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


class TableExtractor:
    """
    Extracts tables from PDFs and converts them to LangChain Documents.

    Uses pdfplumber for robust table detection and extraction, then
    converts each table to a Markdown representation suitable for
    embedding and LLM reasoning.

    Usage:
        extractor = TableExtractor()
        table_docs = extractor.extract_from_pdf(Path("2023 Q3 AAPL.pdf"))
    """

    def extract_from_pdf(self, pdf_path: Path) -> list[Document]:
        """
        Extract all tables from a single PDF.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            List of Documents, each containing one table in Markdown format.
            Metadata includes source, page_number, table_index, and parsed fields.
        """
        documents = []
        file_metadata = _extract_metadata_from_filename(pdf_path.name)

        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_idx, page in enumerate(pdf.pages):
                    page_number = page_idx + 1
                    tables = page.extract_tables()

                    for table_idx, table in enumerate(tables):
                        # Convert to Markdown with source context
                        source_ctx = (
                            f"{file_metadata.get('ticker', '?')} "
                            f"{file_metadata.get('year', '?')} "
                            f"{file_metadata.get('quarter', '?')} - "
                            f"Page {page_number}, Table {table_idx + 1}"
                        )
                        markdown = _table_to_markdown(table, source_context=source_ctx)

                        if markdown:
                            metadata = {
                                "source": str(pdf_path),
                                "page_number": page_number,
                                "table_index": table_idx,
                                "content_type": "table",
                                **file_metadata,
                            }

                            documents.append(
                                Document(page_content=markdown, metadata=metadata)
                            )

            logger.debug(
                f"Extracted {len(documents)} tables from {pdf_path.name}"
            )

        except Exception as e:
            logger.error(f"Table extraction failed for {pdf_path.name}: {e}")

        return documents

    def extract_from_directory(self, directory: Path = None) -> list[Document]:
        """
        Extract tables from all PDFs in a directory.

        Args:
            directory: Path to directory with PDFs. Defaults to settings.pdf_dir.

        Returns:
            Combined list of table Documents from all PDFs.
        """
        directory = directory or settings.pdf_dir
        pdf_files = sorted(directory.glob("*.pdf"))
        all_tables = []

        for pdf_path in pdf_files:
            tables = self.extract_from_pdf(pdf_path)
            all_tables.extend(tables)

        logger.info(
            f"Total tables extracted: {len(all_tables)} from {len(pdf_files)} PDFs"
        )
        return all_tables
