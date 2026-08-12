import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb

from src.config import TOP_K, RERANK_CANDIDATES, RERANK_ENABLED, CHROMA_DIR
from src.ingestion.vector_store import get_collection
from src.retrieval.bm25 import BM25Index, reciprocal_rank_fusion
from src.logger import get_logger

log = get_logger("retriever")

_IMAGE_STORE_DIR = CHROMA_DIR / "images"


def _load_image_b64(chunk_id: str) -> Optional[str]:
    """Load image base64 from disk if it exists."""
    img_path = _IMAGE_STORE_DIR / f"{chunk_id}.b64"
    if img_path.exists():
        return img_path.read_text()
    return None


class Retriever:
    def __init__(self):
        log.info("Loading ChromaDB collection...")
        self.collection: chromadb.Collection = get_collection()
        log.info(f"Collection loaded | vectors={self.collection.count()}")
        self._bm25_index: Optional[BM25Index] = None
        self._bm25_doc_count: int = 0
        self._build_bm25_index()

    def _build_bm25_index(self):
        """Build/rebuild BM25 index from all documents in the collection."""
        count = self.collection.count()
        if count == 0:
            log.warning("Collection is empty — BM25 index skipped")
            return

        # Only rebuild if collection size changed
        if self._bm25_index and self._bm25_doc_count == count:
            return

        log.info(f"Building BM25 index over {count} documents...")
        t0 = time.perf_counter()

        # Pull all documents from ChromaDB
        all_data = self.collection.get(include=["documents", "metadatas"])

        documents = []
        for doc_id, doc_text, meta in zip(all_data["ids"], all_data["documents"], all_data["metadatas"]):
            if doc_text and doc_text.strip():
                documents.append({
                    "id": doc_id,
                    "text": doc_text,
                    "source": meta.get("source", ""),
                    "page": meta.get("page", 0),
                    "chunk_type": meta.get("chunk_type", "text"),
                    "has_image": meta.get("has_image", "false") == "true",
                })

        self._bm25_index = BM25Index(documents)
        self._bm25_doc_count = count
        log.info(f"BM25 index ready in {(time.perf_counter()-t0)*1000:.1f}ms")

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K,
        source_filter: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        # Determine how many candidates to retrieve
        retrieve_k = RERANK_CANDIDATES if RERANK_ENABLED else top_k

        # ── Vector search ─────────────────────────────────────────────────────
        where = None
        if source_filter and len(source_filter) == 1:
            where = {"source": {"$eq": source_filter[0]}}
        elif source_filter and len(source_filter) > 1:
            where = {"source": {"$in": source_filter}}

        kwargs: Dict[str, Any] = {
            "query_texts": [query],
            "n_results": retrieve_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        log.debug(f"Vector search | top_k={retrieve_k}")
        t0 = time.perf_counter()
        results = self.collection.query(**kwargs)
        vector_elapsed = (time.perf_counter() - t0) * 1000

        vector_chunks = []
        for doc_id, doc, meta, dist in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunk: Dict[str, Any] = {
                "text": doc,
                "source": meta.get("source", ""),
                "page": meta.get("page", 0),
                "chunk_type": meta.get("chunk_type", "text"),
                "score": round(1 - dist, 4),
            }
            # Load image from disk if this chunk has one
            if meta.get("has_image") == "true":
                image_b64 = _load_image_b64(doc_id)
                if image_b64:
                    chunk["image_b64"] = image_b64
            vector_chunks.append(chunk)

        log.debug(f"Vector search done in {vector_elapsed:.1f}ms | {len(vector_chunks)} results")

        # ── BM25 search ───────────────────────────────────────────────────────
        if self._bm25_index and self._bm25_index.doc_count > 0:
            t1 = time.perf_counter()
            bm25_results = self._bm25_index.search(query, top_k=retrieve_k)

            # Apply source filter to BM25 results
            if source_filter:
                bm25_results = [r for r in bm25_results if r["source"] in source_filter]

            # Normalize BM25 results to match vector chunk format
            bm25_chunks = []
            for r in bm25_results:
                chunk = {
                    "text": r["text"],
                    "source": r["source"],
                    "page": r["page"],
                    "chunk_type": r["chunk_type"],
                    "score": round(r["bm25_score"] / (r["bm25_score"] + 1), 4),  # Normalize to 0-1
                }
                # Load image from disk if this chunk has one
                if r.get("has_image"):
                    image_b64 = _load_image_b64(r["id"])
                    if image_b64:
                        chunk["image_b64"] = image_b64
                bm25_chunks.append(chunk)

            bm25_elapsed = (time.perf_counter() - t1) * 1000
            log.debug(f"BM25 search done in {bm25_elapsed:.1f}ms | {len(bm25_chunks)} results")

            # ── Reciprocal Rank Fusion ────────────────────────────────────────
            fused = reciprocal_rank_fusion(vector_chunks, bm25_chunks)
            log.debug(f"Hybrid search: {len(vector_chunks)} vector + {len(bm25_chunks)} BM25 → {len(fused)} fused")

            # Take top retrieve_k from fused results
            return fused[:retrieve_k]
        else:
            log.debug("BM25 index unavailable — using vector-only search")
            return vector_chunks
