"""
LLM-based reranker using Ollama.

Strategy: Ask the LLM to score each chunk's relevance to the query on a 0-10 scale.
This acts as a cross-encoder-style reranker without needing a dedicated reranking model.
We over-retrieve (e.g., top_k=20) then rerank down to the final top_k (e.g., 8).
"""
import time
import httpx
from typing import List, Dict, Any

from src.config import OLLAMA_BASE_URL, LLM_MODEL
from src.logger import get_logger

log = get_logger("reranker")

_RERANK_PROMPT = """\
Rate the relevance of the following passage to the query on a scale of 0-10.
Only output a single integer number (0-10). No explanation.

Query: {query}

Passage: {passage}

Relevance score (0-10):"""


def rerank_chunks(
    query: str,
    chunks: List[Dict[str, Any]],
    top_k: int = 8,
) -> List[Dict[str, Any]]:
    """
    Rerank retrieved chunks using LLM-as-judge scoring.
    Returns top_k chunks sorted by relevance score (highest first).
    """
    if len(chunks) <= top_k:
        log.debug(f"Skipping rerank — only {len(chunks)} chunks (≤ top_k={top_k})")
        return chunks

    log.info(f"Reranking {len(chunks)} chunks down to {top_k}...")
    t0 = time.perf_counter()

    scored_chunks = []
    for i, chunk in enumerate(chunks):
        # Truncate passage to avoid excessive token usage
        passage = chunk["text"][:500]
        prompt = _RERANK_PROMPT.format(query=query, passage=passage)

        try:
            resp = httpx.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": LLM_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.0, "num_predict": 5},
                },
                timeout=30.0,
            )
            resp.raise_for_status()
            raw_score = resp.json()["response"].strip()

            # Parse score — extract first integer found
            score = _parse_score(raw_score)
            log.debug(f"  chunk {i+1}: score={score} (raw='{raw_score}') [{chunk['source']} p{chunk['page']}]")

        except Exception as e:
            log.warning(f"  chunk {i+1}: rerank failed ({e}), using retrieval score")
            # Fall back to the original retrieval score (0-1 scale → 0-10)
            score = int(chunk.get("score", 0.5) * 10)

        scored_chunks.append({**chunk, "rerank_score": score})

    # Sort by rerank score (descending), break ties with original retrieval score
    scored_chunks.sort(key=lambda c: (c["rerank_score"], c.get("score", 0)), reverse=True)

    elapsed = (time.perf_counter() - t0) * 1000
    log.info(
        f"Reranking done in {elapsed:.1f}ms | "
        f"top scores: {[c['rerank_score'] for c in scored_chunks[:top_k]]}"
    )

    return scored_chunks[:top_k]


def _parse_score(raw: str) -> int:
    """Extract an integer 0-10 from LLM output. Defaults to 5 on parse failure."""
    # Try to find a number in the response
    for token in raw.split():
        cleaned = token.strip(".,;:!?()[]")
        try:
            val = int(cleaned)
            return max(0, min(10, val))
        except ValueError:
            continue
    # Try float
    try:
        val = int(float(raw.strip()))
        return max(0, min(10, val))
    except (ValueError, TypeError):
        pass
    log.debug(f"Could not parse rerank score from: '{raw}', defaulting to 5")
    return 5
