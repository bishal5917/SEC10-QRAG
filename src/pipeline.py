import time
from typing import List, Dict, Any, Optional

from src.retrieval.retriever import Retriever
from src.generation.llm import generate_answer
from src.config import TOP_K
from src.logger import get_logger

log = get_logger("pipeline")


class RAGPipeline:
    def __init__(self):
        log.info("Initializing retriever...")
        self.retriever = Retriever()
        log.info("Retriever ready")

    def query(
        self,
        question: str,
        top_k: int = TOP_K,
        source_filter: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        log.info(f"[1/3] Retrieving chunks | question=\"{question[:80]}\" top_k={top_k} filter={source_filter}")
        t0 = time.perf_counter()
        chunks = self.retriever.retrieve(question, top_k=top_k, source_filter=source_filter)
        log.info(f"[1/3] Retrieved {len(chunks)} chunks in {(time.perf_counter()-t0)*1000:.1f}ms")

        if not chunks:
            log.warning("No relevant chunks found for query")
            return {"answer": "No relevant documents found for this query.", "sources": [], "chunks": []}

        for i, c in enumerate(chunks, 1):
            log.debug(f"  chunk {i}: [{c['chunk_type'].upper()}] {c['source']} p{c['page']} score={c['score']}")

        log.info(f"[2/3] Generating answer with {len(chunks)} chunks...")
        t1 = time.perf_counter()
        answer = generate_answer(question, chunks)
        log.info(f"[2/3] Answer generated in {(time.perf_counter()-t1)*1000:.1f}ms")

        unique_sources = sorted({c["source"] for c in chunks})
        log.info(f"[3/3] Done | sources={unique_sources}")

        return {
            "answer": answer,
            "sources": unique_sources,
            "chunks": [
                {
                    "source": c["source"],
                    "page": c["page"],
                    "chunk_type": c["chunk_type"],
                    "score": c["score"],
                    "text_preview": c["text"][:200],
                }
                for c in chunks
            ],
        }
