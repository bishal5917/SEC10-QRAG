"""
BM25 keyword search for hybrid retrieval.

Uses rank_bm25 to score documents by term frequency, then fuses results
with vector similarity scores using Reciprocal Rank Fusion (RRF).
"""
import math
import re
import time
from typing import List, Dict, Any, Optional

from src.logger import get_logger

log = get_logger("bm25")


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer with lowercasing."""
    text = text.lower()
    # Split on non-alphanumeric, keep tokens of length >= 2
    tokens = re.findall(r"[a-z0-9]+", text)
    return [t for t in tokens if len(t) >= 2]


class BM25Index:
    """
    Lightweight BM25 index built over document texts stored in ChromaDB.
    Rebuilt on init by pulling all documents from the collection.
    """

    def __init__(self, documents: List[Dict[str, Any]]):
        """
        Build BM25 index from a list of documents.
        Each doc must have 'id', 'text', and 'metadata' keys.
        """
        t0 = time.perf_counter()
        self.documents = documents
        self.corpus = [_tokenize(doc["text"]) for doc in documents]
        self.doc_count = len(self.corpus)

        if self.doc_count == 0:
            self.avgdl = 0
            self.idf = {}
            log.warning("BM25 index built with 0 documents")
            return

        # Compute IDF for each term
        self.avgdl = sum(len(doc) for doc in self.corpus) / self.doc_count
        self.idf = self._compute_idf()

        elapsed = (time.perf_counter() - t0) * 1000
        log.info(f"BM25 index built: {self.doc_count} docs, {len(self.idf)} terms in {elapsed:.1f}ms")

    def _compute_idf(self) -> Dict[str, float]:
        """Compute inverse document frequency for all terms."""
        df = {}  # document frequency
        for doc_tokens in self.corpus:
            seen = set(doc_tokens)
            for token in seen:
                df[token] = df.get(token, 0) + 1

        idf = {}
        for term, freq in df.items():
            # Standard BM25 IDF formula
            idf[term] = math.log((self.doc_count - freq + 0.5) / (freq + 0.5) + 1.0)
        return idf

    def search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Score all documents against query using BM25.
        Returns top_k results with score added.
        """
        if self.doc_count == 0:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        k1 = 1.5
        b = 0.75

        scores = []
        for i, doc_tokens in enumerate(self.corpus):
            score = 0.0
            doc_len = len(doc_tokens)

            # Count term frequencies in this doc
            tf_map = {}
            for token in doc_tokens:
                tf_map[token] = tf_map.get(token, 0) + 1

            for q_token in query_tokens:
                if q_token not in self.idf:
                    continue
                tf = tf_map.get(q_token, 0)
                if tf == 0:
                    continue
                idf = self.idf[q_token]
                # BM25 scoring formula
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * doc_len / self.avgdl)
                score += idf * (numerator / denominator)

            scores.append((i, score))

        # Sort by score descending, filter zero scores
        scores = [(i, s) for i, s in scores if s > 0]
        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for i, bm25_score in scores[:top_k]:
            doc = self.documents[i].copy()
            doc["bm25_score"] = round(bm25_score, 4)
            results.append(doc)

        return results


def reciprocal_rank_fusion(
    vector_results: List[Dict[str, Any]],
    bm25_results: List[Dict[str, Any]],
    k: int = 60,
    vector_weight: float = 0.6,
    bm25_weight: float = 0.4,
) -> List[Dict[str, Any]]:
    """
    Combine vector and BM25 results using Reciprocal Rank Fusion.
    
    RRF score = weight * (1 / (k + rank))
    
    Args:
        vector_results: Chunks from vector similarity search (must have 'text' key)
        bm25_results: Chunks from BM25 search (must have 'text' key)
        k: RRF constant (higher = more equal weighting across ranks)
        vector_weight: Weight for vector results
        bm25_weight: Weight for BM25 results
    
    Returns:
        Fused results sorted by combined RRF score
    """
    # Build score map keyed by text content (since IDs may differ)
    rrf_scores: Dict[str, float] = {}
    chunk_map: Dict[str, Dict[str, Any]] = {}

    # Score vector results
    for rank, chunk in enumerate(vector_results, 1):
        key = chunk["text"][:200]  # Use text prefix as dedup key
        rrf_scores[key] = rrf_scores.get(key, 0) + vector_weight * (1.0 / (k + rank))
        if key not in chunk_map:
            chunk_map[key] = chunk

    # Score BM25 results
    for rank, chunk in enumerate(bm25_results, 1):
        key = chunk["text"][:200]
        rrf_scores[key] = rrf_scores.get(key, 0) + bm25_weight * (1.0 / (k + rank))
        if key not in chunk_map:
            chunk_map[key] = chunk

    # Sort by fused score
    sorted_keys = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)

    results = []
    for key in sorted_keys:
        chunk = chunk_map[key].copy()
        chunk["rrf_score"] = round(rrf_scores[key], 6)
        results.append(chunk)

    log.debug(f"RRF fusion: {len(vector_results)} vector + {len(bm25_results)} BM25 → {len(results)} fused")
    return results
