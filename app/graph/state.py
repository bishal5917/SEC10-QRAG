"""
LangGraph State Definitions.

Defines the typed state schema for the RAG workflow graph.
The state flows through each node, accumulating data as the
query progresses through retrieval, grading, and generation.

LangGraph uses TypedDict to define what data flows between nodes,
providing type safety and clear data contracts.
"""

from typing import Optional

from langchain_core.documents import Document
from typing_extensions import TypedDict

from app.retrieval.multimodal_retriever import RetrievalResult


class RAGState(TypedDict):
    """
    State schema for the multimodal RAG workflow graph.

    Each field represents a piece of data that flows through the
    graph nodes. Fields are populated progressively as the query
    moves through: retrieve → grade → generate.

    Fields:
        question: The user's original natural language query.
        metadata_filter: Optional filter to narrow retrieval (e.g., ticker).
        retrieval_result: Full multi-modal retrieval output.
        relevant_text_docs: Text documents that passed relevance grading.
        relevant_table_docs: Table documents that passed relevance grading.
        generation: The final generated answer string.
        include_images: Whether to include images in generation.
        error: Error message if any step fails.
    """

    # ─── Input ────────────────────────────────────────────────────────────
    question: str
    metadata_filter: Optional[dict]
    include_images: bool
    skip_grading: bool

    # ─── Retrieval Output ─────────────────────────────────────────────────
    retrieval_result: Optional[RetrievalResult]

    # ─── Grading Output ───────────────────────────────────────────────────
    relevant_text_docs: Optional[list[Document]]
    relevant_table_docs: Optional[list[Document]]

    # ─── Generation Output ────────────────────────────────────────────────
    generation: Optional[str]

    # ─── Error Handling ───────────────────────────────────────────────────
    error: Optional[str]
