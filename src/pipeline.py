import time
from typing import List, Dict, Any, Optional

from src.retrieval.retriever import Retriever
from src.retrieval.reranker import rerank_chunks
from src.generation.llm import generate_answer
from src.config import TOP_K, RERANK_ENABLED, RERANK_CANDIDATES
from src.logger import get_logger

log = get_logger("pipeline")


class RAGPipeline:
    def __init__(self):
        log.info("Initializing retriever...")
        self.retriever = Retriever()
        log.info(f"Retriever ready | rerank={'ON' if RERANK_ENABLED else 'OFF'}")

    def query(
        self,
        question: str,
        top_k: int = TOP_K,
        source_filter: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        # Over-retrieve if reranking is enabled
        retrieve_k = RERANK_CANDIDATES if RERANK_ENABLED else top_k

        log.info(f"[1/4] Retrieving chunks | question=\"{question[:80]}\" retrieve_k={retrieve_k} filter={source_filter}")
        t0 = time.perf_counter()
        chunks = self.retriever.retrieve(question, top_k=retrieve_k, source_filter=source_filter)
        log.info(f"[1/4] Retrieved {len(chunks)} chunks in {(time.perf_counter()-t0)*1000:.1f}ms")

        if not chunks:
            log.warning("No relevant chunks found for query")
            return {"answer": "No relevant documents found for this query.", "sources": [], "chunks": []}

        # Rerank
        if RERANK_ENABLED and len(chunks) > top_k:
            log.info(f"[2/4] Reranking {len(chunks)} → {top_k} chunks...")
            t1 = time.perf_counter()
            chunks = rerank_chunks(question, chunks, top_k=top_k)
            log.info(f"[2/4] Reranking done in {(time.perf_counter()-t1)*1000:.1f}ms")
        else:
            chunks = chunks[:top_k]
            log.info("[2/4] Reranking skipped")

        for i, c in enumerate(chunks, 1):
            score_info = f"score={c['score']}"
            if "rerank_score" in c:
                score_info += f" rerank={c['rerank_score']}"
            log.debug(f"  chunk {i}: [{c['chunk_type'].upper()}] {c['source']} p{c['page']} {score_info}")

        log.info(f"[3/4] Generating answer with {len(chunks)} chunks...")
        t2 = time.perf_counter()
        answer = generate_answer(question, chunks)
        log.info(f"[3/4] Answer generated in {(time.perf_counter()-t2)*1000:.1f}ms")

        unique_sources = sorted({c["source"] for c in chunks})
        log.info(f"[4/4] Done | sources={unique_sources}")

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
