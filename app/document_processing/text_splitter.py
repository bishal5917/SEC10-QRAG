"""
Text Splitting Module — Layout-Aware (Document-Specific) Chunking.

Splits extracted 10-Q text at DOCUMENT STRUCTURE boundaries first (section
notes, statement headings, MD&A items), then falls back to recursive
paragraph/sentence splitting for size control.

Why layout-aware:
    10-Q filings have a clear logical structure ("Note 7: Income Taxes",
    "Item 2. Management's Discussion...", statement titles). Plain recursive
    splitting is size-driven and structure-blind — it can orphan a section
    heading at the tail of an unrelated chunk (e.g. "Note 7: Income Taxes"
    stuck onto acquisition text), which dilutes the chunk's embedding and
    hurts retrieval. Splitting at these markers keeps each section coherent
    (heading + its numbers + explanation stay together).

Strategy:
    1. Primary: break at section-heading patterns (regex separators), placed
       FIRST in the separator hierarchy so the splitter prefers them.
       keep_separator="start" keeps the heading attached to the text that
       FOLLOWS it (so "Note 7:" leads its section rather than ending the
       previous chunk).
    2. Fallback: standard recursive separators (paragraph → line → sentence →
       word → char) for any section still larger than chunk_size.
    3. Tiny fragments below a minimum size are merged into the next chunk so
       an over-eager heading match can't create meaningless slivers.

Note: This affects the TEXT path only. Tables are handled separately by the
table extractor (pdfplumber → Markdown), unchanged.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def get_text_splitter() -> RecursiveCharacterTextSplitter:
    """
    Create a layout-aware recursive text splitter.

    The separator hierarchy is: [section-heading regexes] + [recursive defaults].
    Section headings come first so the splitter prefers to break there; when a
    section is bigger than chunk_size it falls back to the standard boundaries.

    Returns:
        Configured RecursiveCharacterTextSplitter (regex separators enabled).
    """
    # Section-heading patterns (regex). Configurable via settings so they can
    # be tuned per document set. These mark natural 10-Q section boundaries.
    section_separators = list(settings.section_heading_patterns)

    # Standard recursive fallbacks (used within oversized sections).
    fallback_separators = ["\n\n", "\n", ". ", " ", ""]

    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
        separators=section_separators + fallback_separators,
        is_separator_regex=True,          # section separators are regex patterns
        keep_separator="start",           # keep the heading with the text that follows it
    )


def _merge_tiny_chunks(chunks: list[Document], min_size: int) -> list[Document]:
    """
    Merge chunks smaller than min_size into the following chunk.

    An over-eager section-heading match can create tiny slivers (e.g. just a
    heading line). Merging them forward keeps each stored chunk meaningful.

    Args:
        chunks: Chunked Documents (in order).
        min_size: Minimum acceptable chunk length in characters.

    Returns:
        List of Documents with tiny fragments merged forward.
    """
    if not chunks:
        return chunks

    merged: list[Document] = []
    carry = None  # a pending tiny chunk to prepend to the next one

    for chunk in chunks:
        if carry is not None:
            # Prepend the carried tiny text to this chunk (keep this chunk's metadata)
            chunk = Document(
                page_content=carry.page_content + "\n" + chunk.page_content,
                metadata=chunk.metadata,
            )
            carry = None

        if len(chunk.page_content) < min_size:
            carry = chunk  # too small — carry it forward to merge into the next
        else:
            merged.append(chunk)

    # If a tiny chunk is left at the very end, append it to the last chunk
    if carry is not None:
        if merged:
            last = merged[-1]
            merged[-1] = Document(
                page_content=last.page_content + "\n" + carry.page_content,
                metadata=last.metadata,
            )
        else:
            merged.append(carry)

    return merged


def split_documents(documents: list[Document]) -> list[Document]:
    """
    Split page Documents into layout-aware chunks.

    Splits at section headings first (keeping headings with their content),
    falls back to recursive splitting for size, then merges tiny fragments.
    All source metadata (page, ticker, quarter) is preserved.

    Args:
        documents: Full-page Documents from the PDF loader.

    Returns:
        List of chunked Documents with preserved and enriched metadata.
    """
    splitter = get_text_splitter()

    chunks = splitter.split_documents(documents)

    # Merge tiny slivers (e.g. lone headings) into neighbours
    chunks = _merge_tiny_chunks(chunks, min_size=settings.min_chunk_size)

    # Add chunk_index / content_type metadata
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = idx
        chunk.metadata["content_type"] = "text"

    logger.info(
        f"Split {len(documents)} documents into {len(chunks)} layout-aware chunks "
        f"(chunk_size={settings.chunk_size}, overlap={settings.chunk_overlap})"
    )

    return chunks
