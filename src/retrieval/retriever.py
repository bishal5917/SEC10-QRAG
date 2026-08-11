import time
from typing import List, Dict, Any, Optional
import chromadb

from src.config import TOP_K
from src.ingestion.vector_store import get_collection
from src.logger import get_logger

log = get_logger("retriever")


class Retriever:
    def __init__(self):
        log.info("Loading ChromaDB collection...")
        self.collection: chromadb.Collection = get_collection()
        log.info(f"Collection loaded | vectors={self.collection.count()}")

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K,
        source_filter: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        where = None
        if source_filter and len(source_filter) == 1:
            where = {"source": {"$eq": source_filter[0]}}
        elif source_filter and len(source_filter) > 1:
            where = {"source": {"$in": source_filter}}

        if where:
            log.debug(f"Applying source filter: {where}")

        kwargs: Dict[str, Any] = {
            "query_texts": [query],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        log.debug(f"Querying ChromaDB | top_k={top_k}")
        t0 = time.perf_counter()
        results = self.collection.query(**kwargs)
        log.debug(f"ChromaDB query done in {(time.perf_counter()-t0)*1000:.1f}ms")

        chunks = []
        for doc, meta, dist in zip(
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
            if meta.get("image_b64"):
                chunk["image_b64"] = meta["image_b64"]
            chunks.append(chunk)

        log.debug(f"Top scores: {[c['score'] for c in chunks]}")
        return chunks
