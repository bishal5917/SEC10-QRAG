"""
Retrieval Evaluation Metrics.

Document-level metrics that measure whether *retrieval* found the right
source documents, independent of the final answer quality. Plus a
Negative Rejection metric for "no information" questions.

Why these exist:
    Answer-quality metrics (BLEU, semantic similarity, judge) measure the
    final output. But a RAG system can produce a good answer with poor
    retrieval, or a bad answer despite good retrieval. These metrics isolate
    the retrieval step so you can diagnose where problems actually are.

Metrics:
    1. Hit Rate   — did retrieval include a chunk from the correct source? (0/1)
    2. MRR        — how high did the first correct-source chunk rank? (0-1)
    3. Negative Rejection (R-Rate) — for "no info" reference answers, did the
       system correctly decline instead of fabricating? (0/1, or None if N/A)

Ground truth:
    Parsed from the CSV's "Source Docs" column, which uses "*...*" markers:
        "*AAPL*"           → any AAPL document (ticker-level match)
        "*2023 Q3 INTC*"   → a specific document (year + quarter + ticker)
    Matching is therefore at the DOCUMENT level (not chunk level), since that
    is the granularity the ground truth provides.
"""

import re

from langchain_core.documents import Document

from app.core.logging import get_logger

logger = get_logger(__name__)


# Known tickers for parsing the Source Docs field
_KNOWN_TICKERS = {"AAPL", "AMZN", "INTC", "MSFT", "NVDA"}


def parse_expected_sources(source_docs: str) -> list[dict]:
    """
    Parse the CSV "Source Docs" field into structured match criteria.

    Handles the "*...*" marker format:
        "*AAPL*"          → [{"ticker": "AAPL"}]                  (any AAPL doc)
        "*2023 Q3 INTC*"  → [{"ticker": "INTC", "year": "2023", "quarter": "Q3"}]
    Multiple markers in one field are all returned.

    Args:
        source_docs: Raw value from the "Source Docs" column.

    Returns:
        List of criteria dicts. Each dict is a set of metadata fields that a
        retrieved document must match to count as a "correct" source.
    """
    if not source_docs:
        return []

    criteria = []

    # Extract each "*...*" group; fall back to the whole string if no markers
    groups = re.findall(r"\*([^*]+)\*", source_docs)
    if not groups:
        groups = [source_docs]

    for group in groups:
        tokens = group.strip().split()
        crit = {}
        for tok in tokens:
            tok_up = tok.upper()
            if tok_up in _KNOWN_TICKERS:
                crit["ticker"] = tok_up
            elif re.fullmatch(r"20\d{2}", tok):
                crit["year"] = tok
            elif re.fullmatch(r"[Qq][1-4]", tok):
                crit["quarter"] = tok_up
        if crit:
            criteria.append(crit)

    return criteria


def _doc_matches(doc: Document, criteria: dict) -> bool:
    """
    Check whether a retrieved document satisfies one criteria dict.

    A document matches only if ALL fields in the criteria match its metadata
    (e.g., {"ticker": "INTC", "quarter": "Q3"} requires both to match).

    Args:
        doc: A retrieved LangChain Document.
        criteria: One criteria dict from parse_expected_sources().

    Returns:
        True if the document matches all fields in the criteria.
    """
    meta = doc.metadata
    for key, expected in criteria.items():
        actual = str(meta.get(key, "")).upper()
        if actual != str(expected).upper():
            return False
    return True


def _doc_is_correct(doc: Document, expected_criteria: list[dict]) -> bool:
    """Return True if the doc matches ANY of the expected criteria."""
    return any(_doc_matches(doc, crit) for crit in expected_criteria)


def hit_rate(retrieved_docs: list[Document], source_docs: str) -> float:
    """
    Hit Rate: did retrieval include at least one correct-source document?

    Args:
        retrieved_docs: Documents retrieved for the query (text + tables).
        source_docs: Raw "Source Docs" ground-truth string.

    Returns:
        1.0 if any retrieved doc matches an expected source, else 0.0.
        Returns None if the ground truth can't be parsed.
    """
    expected = parse_expected_sources(source_docs)
    if not expected:
        return None

    for doc in retrieved_docs:
        if _doc_is_correct(doc, expected):
            return 1.0
    return 0.0


def mrr(retrieved_docs: list[Document], source_docs: str) -> float:
    """
    Mean Reciprocal Rank: 1 / (rank of the first correct-source document).

    Documents are considered in the order they appear in retrieved_docs
    (best-ranked first, e.g. after re-ranking). If the first correct doc is
    at position 1 → 1.0; position 3 → 0.333; not found → 0.0.

    Args:
        retrieved_docs: Ordered documents retrieved for the query.
        source_docs: Raw "Source Docs" ground-truth string.

    Returns:
        Reciprocal rank in [0, 1]. Returns None if ground truth unparseable.
    """
    expected = parse_expected_sources(source_docs)
    if not expected:
        return None

    for rank, doc in enumerate(retrieved_docs, start=1):
        if _doc_is_correct(doc, expected):
            return round(1.0 / rank, 4)
    return 0.0


# ─── Negative Rejection (R-Rate) ──────────────────────────────────────────────

# Phrases indicating the reference answer declines / says no info is available
_NO_INFO_SIGNALS = [
    "no explicit details",
    "no information",
    "do not contain",
    "does not contain",
    "cannot be determined",
    "could not be determined",
    "not available",
    "no specific",
    "not provided",
    "no notable",
]


def _is_no_info(text: str) -> bool:
    """Check if an answer declines / indicates no information is available."""
    low = (text or "").lower()
    return any(sig in low for sig in _NO_INFO_SIGNALS)


def negative_rejection(generated: str, reference: str):
    """
    Negative Rejection (R-Rate): for questions where the correct answer is
    "no information available", did the system correctly decline?

    Only applies when the REFERENCE answer itself indicates no info. For all
    other questions this returns None (not applicable) so it doesn't skew the
    average.

    Args:
        generated: The generated answer text.
        reference: The ground-truth reference answer.

    Returns:
        1.0 if reference says "no info" AND generated also declines,
        0.0 if reference says "no info" but generated fabricated an answer,
        None if this question isn't a negative-rejection case.
    """
    if not _is_no_info(reference):
        return None  # Not a negative-rejection question

    return 1.0 if _is_no_info(generated) else 0.0


def compute_retrieval_metrics(
    retrieved_docs: list[Document],
    source_docs: str,
    generated: str,
    reference: str,
) -> dict:
    """
    Compute all retrieval-side metrics for one Q&A pair.

    Args:
        retrieved_docs: Ordered documents used for generation (text + tables).
        source_docs: Raw "Source Docs" ground-truth string.
        generated: The generated answer text.
        reference: The ground-truth reference answer.

    Returns:
        Dict with hit_rate, mrr, and negative_rejection.
    """
    return {
        "hit_rate": hit_rate(retrieved_docs, source_docs),
        "mrr": mrr(retrieved_docs, source_docs),
        "negative_rejection": negative_rejection(generated, reference),
    }
