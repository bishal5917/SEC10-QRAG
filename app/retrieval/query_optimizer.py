"""
Query Optimization Module.

Rewrites/expands a user's question into a retrieval-optimized query before
embedding, to bridge the vocabulary gap between how questions are phrased
and how the source documents are worded.

Motivating example:
    Question:  "How does Amazon's revenue distribution across segments like
                e-commerce and AWS compare to costs?"
    Problem:   The actual 10-Q segment table uses "North America / International
               / AWS" and "operating expenses" — none of the query's words.
               So embedding/reranking ranks the real table low.
    Fix:       An LLM rewrites the query to include the document's likely
               vocabulary (segment names, "net sales", "operating income"),
               so retrieval surfaces the right table.

Design:
    - LLM-based rewrite using Gemini (one extra API call per query).
    - The rewritten query is used ONLY for retrieval (embedding/search).
      The ORIGINAL question is still what the answer is generated for, so the
      user's intent and the answer format are unaffected.
    - Controlled via config: query_optimization = "none" | "llm".
    - Fails open: if the rewrite call errors, we fall back to the original query.
"""

from google import genai
from google.genai import types

from app.core.config import settings
from app.core.logging import get_logger
from app.core.observability import observe

logger = get_logger(__name__)


# ─── Rewrite prompt ───────────────────────────────────────────────────────────
# Instructs the model to produce a search-optimized version of the question,
# enriched with the terminology likely found in financial 10-Q filings.
REWRITE_PROMPT = """You are a search query optimizer for a financial-document retrieval system.
The documents are company 10-Q quarterly filings (income statements, segment tables, notes).

Rewrite the user's question into a single, keyword-rich search query that will
match the terminology actually used in 10-Q filings. Guidelines:
- Expand vague business terms into the specific terms filings use.
  (e.g. "revenue" → "net sales revenue"; "segments" → "reportable segments North America International AWS operating income")
- Keep any company names, quarters, and years from the original.
- Include the key financial line items the question is really asking about.
- Output ONLY the rewritten query as plain text. No explanation, no quotes.

User question: {question}

Rewritten search query:"""


class QueryOptimizer:
    """
    LLM-based query rewriter for improving retrieval.

    Usage:
        optimizer = QueryOptimizer()
        search_query = optimizer.optimize("How has Apple's revenue changed?")
        # → "Apple total net sales revenue by quarter income statement ..."
    """

    def __init__(self):
        """Initialize the Gemini client for query rewriting."""
        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY required for LLM query optimization")

        self.client = genai.Client(api_key=settings.gemini_api_key)
        self.model = settings.gemini_model

    @observe(name="query_rewrite")
    def optimize(self, question: str) -> str:
        """
        Rewrite a question into a retrieval-optimized search query.

        Args:
            question: The user's original natural-language question.

        Returns:
            The rewritten search query, or the original question if the
            rewrite fails (fail-open).
        """
        prompt = REWRITE_PROMPT.format(question=question)

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,          # deterministic rewrite
                    max_output_tokens=256,
                ),
            )

            rewritten = (response.text or "").strip()

            # Fall back to original if the model returned nothing usable
            if not rewritten:
                logger.warning("Query optimizer returned empty; using original query")
                return question

            logger.info(f"🔧 Query rewritten:\n    original: {question[:80]}\n    rewritten: {rewritten[:120]}")
            return rewritten

        except Exception as e:
            logger.warning(f"Query optimization failed, using original query: {e}")
            return question


def optimize_query(question: str) -> str:
    """
    Optimize a query according to the configured strategy.

    Reads settings.query_optimization:
        - "none" → return the question unchanged (no API call)
        - "llm"  → rewrite with the LLM query optimizer

    Args:
        question: The user's original question.

    Returns:
        The query string to use for retrieval.
    """
    strategy = settings.query_optimization

    if strategy == "llm":
        return QueryOptimizer().optimize(question)

    # "none" or unknown → no optimization
    return question
