"""
LangGraph Node Functions.

Each function represents a node in the RAG workflow graph.
Nodes receive the current state, perform their operation, and
return a partial state update (only the fields they modify).

Graph Structure:
    ┌─────────┐     ┌───────┐     ┌──────────┐     ┌──────────┐
    │ Retrieve│────▶│ Grade │────▶│ Prepare  │────▶│ Generate │
    └─────────┘     └───────┘     │ Context  │     └──────────┘
                                  └──────────┘

Node Responsibilities:
    - retrieve: Query the multi-modal vector store
    - grade_documents: Filter retrieved docs for relevance
    - prepare_context: Assemble final context for generation
    - generate: Produce the answer using Gemini
"""

from app.chains.grader import DocumentGrader
from app.chains.generation import GeminiMultiModalGenerator
from app.core.config import settings
from app.core.logging import get_logger
from app.graph.state import RAGState
from app.retrieval.multimodal_retriever import MultiModalRetriever, RetrievalResult
from app.retrieval.reranker import LocalReranker

logger = get_logger(__name__)

# ─── Module-level component references (set during graph construction) ────────
# These are injected by the workflow builder to avoid re-initialization per call.
_retriever: MultiModalRetriever = None
_grader: DocumentGrader = None
_generator: GeminiMultiModalGenerator = None
_reranker: LocalReranker = None


def set_components(
    retriever: MultiModalRetriever,
    grader: DocumentGrader,
    generator: GeminiMultiModalGenerator,
    reranker: LocalReranker = None,
) -> None:
    """
    Inject shared components into the node functions.

    Called once during graph construction to provide the initialized
    retriever, grader, generator, and reranker to all nodes.

    Args:
        retriever: Initialized MultiModalRetriever.
        grader: Initialized DocumentGrader.
        generator: Initialized GeminiMultiModalGenerator.
        reranker: Initialized LocalReranker (cross-encoder).
    """
    global _retriever, _grader, _generator, _reranker
    _retriever = retriever
    _grader = grader
    _generator = generator
    _reranker = reranker


def retrieve(state: RAGState) -> dict:
    """
    Retrieve node — performs multi-modal vector store search.

    Searches text, table, and image collections using the user's
    question. Applies optional metadata filters (e.g., ticker, quarter).

    Input state fields: question, metadata_filter
    Output state fields: retrieval_result

    Args:
        state: Current graph state with the user's question.

    Returns:
        Partial state update with retrieval_result.
    """
    logger.info(f"📥 RETRIEVE: '{state['question']}'")

    try:
        result = _retriever.retrieve(
            query=state["question"],
            metadata_filter=state.get("metadata_filter"),
        )
        return {"retrieval_result": result}

    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        return {
            "retrieval_result": RetrievalResult(query=state["question"]),
            "error": f"Retrieval error: {e}",
        }


def grade_documents(state: RAGState) -> dict:
    """
    Re-rank node — re-scores retrieved documents using a cross-encoder.

    Uses a local cross-encoder model (BGE-reranker) to re-rank documents
    by true relevance to the query. This is more accurate than cosine
    similarity and runs entirely locally (no API calls).

    Falls back to LLM-based grading if --grade flag is set and re-ranker
    is not available.

    Input state fields: retrieval_result, question, skip_grading
    Output state fields: relevant_text_docs, relevant_table_docs

    Args:
        state: Current state with retrieval results.

    Returns:
        Partial state update with re-ranked document lists.
    """
    logger.info("📋 RERANK: Scoring documents with cross-encoder...")

    retrieval_result = state.get("retrieval_result")
    if not retrieval_result or not retrieval_result.has_results:
        return {
            "relevant_text_docs": [],
            "relevant_table_docs": [],
        }

    # Rerank against the RETRIEVAL-OPTIMIZED query (the rewrite), not the raw
    # question. Retrieval used the rewrite, so scoring with the same query keeps
    # the pipeline consistent and (verified) promotes the correct answer chunks.
    # Fall back to the original question if no rewrite is available.
    rerank_query = retrieval_result.search_query or state["question"]

    # Use cross-encoder re-ranking (local, no API calls).
    # Invoke via the reranker's Runnable so the step is captured in LangChain
    # tracing ([chain/start] "CrossEncoderRerank").
    if _reranker:
        rerank_runnable = _reranker.as_runnable()

        # Re-rank text documents
        relevant_text = rerank_runnable.invoke({
            "query": rerank_query,
            "documents": retrieval_result.text_documents,
        })

        # Re-rank table documents (cutoff from config)
        relevant_tables = rerank_runnable.invoke({
            "query": rerank_query,
            "documents": retrieval_result.table_documents,
            "top_n": settings.rerank_top_n_tables,
        })

        logger.info(
            f"   Text: {len(retrieval_result.text_documents)} → {len(relevant_text)} after rerank | "
            f"Tables: {len(retrieval_result.table_documents)} → {len(relevant_tables)} after rerank"
        )
    else:
        # Fallback: skip reranking, pass through as-is
        logger.info("   No reranker available, passing all documents through")
        relevant_text = retrieval_result.text_documents
        relevant_tables = retrieval_result.table_documents

    return {
        "relevant_text_docs": relevant_text,
        "relevant_table_docs": relevant_tables,
    }


def prepare_context(state: RAGState) -> dict:
    """
    Prepare Context node — assembles final RetrievalResult for generation.

    Replaces the raw retrieval result with one containing only the
    graded (relevant) documents. This ensures the generator only
    sees high-quality, relevant context.

    Input state fields: relevant_text_docs, relevant_table_docs, retrieval_result
    Output state fields: retrieval_result (updated with graded docs)

    Args:
        state: Current state with graded documents.

    Returns:
        Partial state update with filtered retrieval_result.
    """
    logger.info("📦 PREPARE: Assembling context for generation...")

    # Build a new RetrievalResult with only the relevant documents
    graded_result = RetrievalResult(
        query=state["question"],
        text_documents=state.get("relevant_text_docs") or [],
        table_documents=state.get("relevant_table_docs") or [],
        # Images pass through ungraded (CLIP retrieval is semantic enough)
        image_documents=(
            state["retrieval_result"].image_documents
            if state.get("retrieval_result")
            else []
        ),
    )

    return {"retrieval_result": graded_result}


def generate(state: RAGState) -> dict:
    """
    Generate node — produces the final answer using Gemini.

    Sends the assembled multimodal context (text + tables + images)
    to Gemini for answer generation. Handles both multimodal and
    text-only generation paths.

    Input state fields: retrieval_result, include_images
    Output state fields: generation

    Args:
        state: Current state with prepared context.

    Returns:
        Partial state update with the generated answer.
    """
    logger.info("🤖 GENERATE: Producing answer with Gemini...")

    retrieval_result = state.get("retrieval_result")

    try:
        answer = _generator.generate(
            retrieval_result=retrieval_result,
            include_images=state.get("include_images", True),
        )
        return {"generation": answer}

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return {
            "generation": f"Failed to generate answer: {e}",
            "error": str(e),
        }
