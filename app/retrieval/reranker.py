"""
Cross-Encoder Re-ranking Module.

Re-ranks retrieved documents using a cross-encoder model that scores
query-document pairs for relevance. This is more accurate than
bi-encoder cosine similarity because the cross-encoder sees both
the query and document together (full attention between them).

Architecture:
    1. Retriever fetches top-K candidates using fast bi-encoder (BGE)
    2. Re-ranker rescores each candidate with a cross-encoder
    3. Top-N highest-scoring documents are passed to generation

    Bi-encoder (retrieval):  query → vec, doc → vec, cosine(q, d)  [fast, approximate]
    Cross-encoder (rerank):  (query, doc) → relevance_score         [slow, precise]

Design Choices:
    - Uses BAAI/bge-reranker-v2-m3 — multilingual, high-quality reranker
    - Runs entirely locally (no API calls, no rate limits)
    - Cross-platform: CUDA (Windows/Linux), MPS (macOS), CPU (fallback)
    - Retrieves more candidates (top-20) then re-ranks to top-5
"""

from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from app.core.config import settings
from app.core.device import get_device
from app.core.logging import get_logger

logger = get_logger(__name__)


class LocalReranker:
    """
    Cross-encoder re-ranker for improving retrieval precision.

    Scores each (query, document) pair with a cross-encoder that has
    full attention between query and document tokens — significantly
    more accurate than cosine similarity from bi-encoders.

    Runs locally on GPU (CUDA/MPS) or CPU. No API calls required.

    Usage:
        reranker = LocalReranker()
        ranked_docs = reranker.rerank(query="Apple revenue", documents=docs, top_n=5)
    """

    def __init__(self, model_name: str = None):
        """
        Initialize the cross-encoder re-ranking model.

        Args:
            model_name: HuggingFace model identifier for the reranker.
                        Defaults to settings.reranker_model.
        """
        self.model_name = model_name or settings.reranker_model
        device = get_device()

        logger.info(f"Loading reranker model: {self.model_name} (device={device})")

        self.model = CrossEncoder(
            self.model_name,
            device=device,
            max_length=512,  # Truncate long docs to 512 tokens for efficiency
        )

        logger.info("Reranker ready")

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_n: int = None,
    ) -> list[Document]:
        """
        Re-rank documents by relevance to the query.

        Scores all documents with the cross-encoder, sorts by score
        descending, and returns the top-N most relevant.

        Args:
            query: The user's question.
            documents: List of candidate Documents to re-rank.
            top_n: Number of top documents to return after re-ranking.
                   Defaults to settings.rerank_top_n.

        Returns:
            List of Documents sorted by relevance (most relevant first),
            truncated to top_n.
        """
        if not documents:
            return []

        top_n = top_n or settings.rerank_top_n

        # Build query-document pairs for cross-encoder scoring
        pairs = [(query, doc.page_content) for doc in documents]

        # Score all pairs (returns array of float scores)
        scores = self.model.predict(pairs, show_progress_bar=False)

        # Attach scores and sort
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Return top-N
        top_docs = [doc for doc, score in scored_docs[:top_n]]

        logger.info(
            f"Reranked {len(documents)} docs → top {len(top_docs)} "
            f"(scores: {scored_docs[0][1]:.3f} to {scored_docs[-1][1]:.3f})"
        )

        return top_docs

    def as_runnable(self, top_n: int = None):
        """
        Expose the reranker as a LangChain Runnable for tracing/composition.

        The returned RunnableLambda expects a dict input:
            {"query": str, "documents": list[Document]}
        and returns the reranked list[Document]. Running it through LangChain
        means the rerank step emits callback events and appears in traces.

        Args:
            top_n: Optional cutoff override; defaults to settings.rerank_top_n.

        Returns:
            A named RunnableLambda wrapping self.rerank.
        """
        def _rerank(inputs: dict) -> list[Document]:
            return self.rerank(
                query=inputs["query"],
                documents=inputs["documents"],
                top_n=inputs.get("top_n", top_n),
            )

        return RunnableLambda(_rerank, name="CrossEncoderRerank")
