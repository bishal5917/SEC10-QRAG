"""
Query Filter Module.

Rule-based extraction of metadata filters from a natural-language query.
Detects which company (ticker), and optionally which quarter/year, a
question is about, and produces a DB-neutral filter that is translated to a
Qdrant filter (to_qdrant_filter) to narrow retrieval to matching documents.

Why this exists:
    Pure semantic retrieval mixes companies and quarters — a question about
    "Intel's Q3 2023 tax rate" also pulls Apple/Microsoft/NVIDIA tax sections
    because they're semantically similar. Filtering by the explicitly named
    company removes that noise, so retrieval only searches relevant documents.

Design:
    - Ticker filter: ALWAYS applied when a company is named (strong, safe signal).
      Multiple companies → match any of them (Qdrant MatchAny).
    - Quarter filter: applied CAUTIOUSLY. Only when the question targets a single
      quarter AND has no multi-quarter language ("over time", "trend", "across",
      etc.), so comparison/trend questions still retrieve all quarters.
    - No API calls — pure regex/string matching.
"""

import re

from app.core.logging import get_logger

logger = get_logger(__name__)


# ─── Company name/ticker → canonical ticker ───────────────────────────────────
# Maps both the ticker and common company names to the ticker stored in metadata.
_TICKER_ALIASES = {
    "AAPL": ["aapl", "apple"],
    "AMZN": ["amzn", "amazon"],
    "INTC": ["intc", "intel"],
    "MSFT": ["msft", "microsoft"],
    "NVDA": ["nvda", "nvidia"],
}

# ─── Phrases that signal a MULTI-quarter question ─────────────────────────────
# If any appear, we do NOT apply a quarter filter (the question spans quarters).
_MULTI_QUARTER_SIGNALS = [
    "over time",
    "across quarters",
    "across the quarters",
    "across reported",
    "over the quarters",
    "over the reported",
    "trend",
    "trends",
    "changed",
    "change over",
    "compare",
    "comparison",
    "fluctuat",       # matches fluctuate / fluctuation
    "quarter over quarter",
    "each quarter",
    "each reported",
    "every quarter",
    "historical",
    "history",
]

# ─── Regex patterns ───────────────────────────────────────────────────────────
_QUARTER_PATTERN = re.compile(r"\bq([1-4])\b", re.IGNORECASE)
_YEAR_PATTERN = re.compile(r"\b(20\d{2})\b")


def _detect_tickers(query_lower: str) -> list[str]:
    """
    Find all companies mentioned in the query.

    Args:
        query_lower: The query text, lowercased.

    Returns:
        List of canonical tickers detected (deduplicated, order-stable).
    """
    found = []
    for ticker, aliases in _TICKER_ALIASES.items():
        for alias in aliases:
            # Word-boundary match so "intel" doesn't match inside another word
            if re.search(rf"\b{re.escape(alias)}\b", query_lower):
                found.append(ticker)
                break  # one alias hit is enough for this ticker
    return found


def _is_multi_quarter_question(query_lower: str) -> bool:
    """
    Check whether the query spans multiple quarters (so we skip quarter filtering).

    Args:
        query_lower: The query text, lowercased.

    Returns:
        True if the question uses multi-quarter / trend / comparison language.
    """
    return any(signal in query_lower for signal in _MULTI_QUARTER_SIGNALS)


def build_metadata_filter(query: str) -> dict | None:
    """
    Detect company/quarter filters from a query as a DB-neutral structure.

    Returns a plain dict describing what was detected, independent of any
    specific vector DB's filter syntax:
        {"tickers": ["INTC"], "quarter": "Q3"}
        {"tickers": ["INTC", "NVDA"], "quarter": None}

    Ticker: always included when detected (one or more companies).
    Quarter: included cautiously — only when exactly one quarter is named and
             the question is not a multi-quarter / trend / comparison question.

    Args:
        query: The user's natural-language question.

    Returns:
        A neutral filter dict, or None if nothing was confidently detected.
        Translate to a specific DB via to_qdrant_filter().
    """
    query_lower = query.lower()

    # Ticker (strong, safe signal)
    tickers = _detect_tickers(query_lower)

    # Quarter (cautious)
    quarter = None
    quarter_matches = _QUARTER_PATTERN.findall(query_lower)
    is_multi = _is_multi_quarter_question(query_lower)
    if len(set(quarter_matches)) == 1 and not is_multi:
        quarter = f"Q{quarter_matches[0]}"

    if not tickers and quarter is None:
        return None

    result = {"tickers": tickers, "quarter": quarter}
    logger.info(f"🔎 Auto-filter: {result}  (multi_quarter={is_multi})")
    return result


def to_qdrant_filter(neutral_filter: dict | None):
    """
    Translate the neutral filter dict into a Qdrant Filter object.

    Qdrant payloads store document metadata under the "metadata" key, so
    fields are addressed as "metadata.ticker", "metadata.quarter".

    Args:
        neutral_filter: Output of build_metadata_filter(), or None.

    Returns:
        A qdrant_client.http.models.Filter, or None if no filter.
    """
    if not neutral_filter:
        return None

    from qdrant_client.http import models

    conditions = []

    tickers = neutral_filter.get("tickers") or []
    if len(tickers) == 1:
        conditions.append(
            models.FieldCondition(
                key="metadata.ticker",
                match=models.MatchValue(value=tickers[0]),
            )
        )
    elif len(tickers) >= 2:
        # Match ANY of the tickers
        conditions.append(
            models.FieldCondition(
                key="metadata.ticker",
                match=models.MatchAny(any=tickers),
            )
        )

    quarter = neutral_filter.get("quarter")
    if quarter:
        conditions.append(
            models.FieldCondition(
                key="metadata.quarter",
                match=models.MatchValue(value=quarter),
            )
        )

    if not conditions:
        return None

    # All conditions must match (ticker AND quarter)
    return models.Filter(must=conditions)
