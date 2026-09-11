"""
LangGraph Workflow Definition.

Defines the RAG query workflow as a directed graph using LangGraph.
This is the central orchestration layer that controls the flow:

    Query → Retrieve → Grade → Prepare Context → Generate → Answer

LangGraph provides:
    - Typed state management across nodes
    - Conditional routing (e.g., skip grading if no results)
    - Visual graph representation for debugging
    - Streaming support for real-time output
    - Built-in error handling and retries

Architecture:
    ┌──────────────────────────────────────────────────────────────┐
    │                    LangGraph RAG Workflow                      │
    │                                                                │
    │  START ──▶ retrieve ──▶ grade_docs ──▶ prepare ──▶ generate   │
    │                            │                          │        │
    │                            ▼                          ▼        │
    │                    (filter irrelevant)          (Gemini LLM)   │
    │                                                       │        │
    │                                                    END ◀───────│
    └──────────────────────────────────────────────────────────────┘
"""

from langgraph.graph import END, StateGraph

from app.chains.generation import GeminiMultiModalGenerator
from app.chains.grader import DocumentGrader
from app.core.config import settings
from app.core.logging import get_logger
from app.embeddings.clip_embeddings import CLIPImageEmbeddings
from app.embeddings.text_embeddings import BGETextEmbeddings
from app.graph import nodes
from app.graph.state import RAGState
from app.retrieval.multimodal_retriever import MultiModalRetriever
from app.vector_store.store import MultiModalVectorStore

logger = get_logger(__name__)


def build_rag_graph(
    vector_store: MultiModalVectorStore,
    text_embeddings: BGETextEmbeddings,
    clip_embeddings: CLIPImageEmbeddings,
) -> StateGraph:
    """
    Build and compile the RAG workflow graph.

    Initializes all components (retriever, grader, generator) and
    wires them into a LangGraph StateGraph with defined edges.

    Args:
        vector_store: Initialized multi-modal vector store.
        text_embeddings: BGE text embedding model.
        clip_embeddings: CLIP image embedding model.

    Returns:
        Compiled LangGraph workflow ready for invocation.
    """
    logger.info("Building LangGraph RAG workflow...")

    # ─── Initialize components ────────────────────────────────────────────
    retriever = MultiModalRetriever(
        vector_store=vector_store,
        text_embeddings=text_embeddings,
        clip_embeddings=clip_embeddings,
    )
    grader = DocumentGrader()
    generator = GeminiMultiModalGenerator()

    # Initialize local cross-encoder reranker only if enabled in settings
    reranker = None
    if settings.enable_reranker:
        from app.retrieval.reranker import LocalReranker
        reranker = LocalReranker()
    else:
        logger.info("Re-ranker disabled (settings.enable_reranker=False)")

    # Inject components into node functions
    nodes.set_components(retriever, grader, generator, reranker)

    # ─── Define the graph ─────────────────────────────────────────────────
    workflow = StateGraph(RAGState)

    # Add nodes (each maps to a function in nodes.py)
    workflow.add_node("retrieve", nodes.retrieve)
    workflow.add_node("rerank", nodes.grade_documents)
    workflow.add_node("prepare_context", nodes.prepare_context)
    workflow.add_node("generate", nodes.generate)

    # ─── Define edges (execution flow) ───────────────────────────────────
    # Entry point: start with retrieval
    workflow.set_entry_point("retrieve")

    # Linear flow: retrieve → rerank → prepare → generate → end
    workflow.add_edge("retrieve", "rerank")
    workflow.add_edge("rerank", "prepare_context")
    workflow.add_edge("prepare_context", "generate")
    workflow.add_edge("generate", END)

    # ─── Compile the graph ───────────────────────────────────────────────
    compiled = workflow.compile()

    logger.info("✓ LangGraph RAG workflow compiled successfully")
    return compiled


class RAGWorkflow:
    """
    High-level wrapper around the LangGraph RAG workflow.

    Provides a clean interface for querying the RAG system without
    needing to manage graph state directly.

    Usage:
        workflow = RAGWorkflow()  # Initializes all models
        answer = workflow.query("What was Apple's revenue in Q3 2023?")
        answer = workflow.query("Compare NVDA and INTC", ticker_filter="NVDA")
    """

    def __init__(self):
        """
        Initialize the RAG workflow with all required models.

        This loads:
            - BGE text embeddings model
            - CLIP image embeddings model
            - Qdrant vector store (embedded, hybrid; connects to existing data)
            - Gemini for grading and generation
        """
        from rich.console import Console
        from rich.progress import Progress, SpinnerColumn, TextColumn

        console = Console()
        console.print("\n[bold blue]Initializing RAG Workflow...[/bold blue]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Loading text embeddings (BGE)...", total=None)
            self.text_embeddings = BGETextEmbeddings()

            progress.update(task, description="Loading image embeddings (CLIP)...")
            self.clip_embeddings = CLIPImageEmbeddings()

            progress.update(task, description="Connecting to vector store...")
            self.vector_store = MultiModalVectorStore(
                text_embeddings=self.text_embeddings,
                image_embeddings=self.clip_embeddings,
            )

            progress.update(task, description="Building LangGraph workflow...")
            self.graph = build_rag_graph(
                vector_store=self.vector_store,
                text_embeddings=self.text_embeddings,
                clip_embeddings=self.clip_embeddings,
            )

        # Display vector store stats
        stats = self.vector_store.get_stats()
        console.print(f"[dim]  Vector store: {stats}[/dim]")
        console.print("[bold green]✓ RAG Workflow ready![/bold green]\n")

    def query(
        self,
        question: str,
        metadata_filter: dict = None,
        include_images: bool = None,
        skip_grading: bool = True,
    ) -> str:
        """
        Execute the full RAG pipeline for a question.

        Runs the LangGraph workflow: retrieve → grade → generate.

        Args:
            question: Natural language question about the documents.
            metadata_filter: Optional dict to filter by metadata
                            (e.g., {"ticker": "AAPL"}).
            include_images: Whether to include page images in generation.
            skip_grading: Skip LLM-based document grading (default True
                         to save API calls on free tier). Set False for
                         higher quality answers at cost of more API calls.

        Returns:
            Generated answer string.
        """
        answer, _ = self.query_with_trace(
            question=question,
            metadata_filter=metadata_filter,
            include_images=include_images,
            skip_grading=skip_grading,
        )
        return answer

    def close(self) -> None:
        """Close underlying resources (embedded Qdrant client) cleanly."""
        try:
            self.vector_store.close()
        except Exception:
            pass

    def query_with_trace(
        self,
        question: str,
        metadata_filter: dict = None,
        include_images: bool = None,
        skip_grading: bool = True,
    ):
        """
        Same as query(), but also returns the retrieval result.

        Useful for evaluation, where we need to inspect which documents
        were retrieved (to compute Hit Rate, MRR, etc.) in addition to
        the generated answer.

        Args:
            question: Natural language question about the documents.
            metadata_filter: Optional dict to filter by metadata.
            include_images: Whether to include page images in generation.
            skip_grading: Skip LLM-based document grading.

        Returns:
            Tuple of (answer_string, RetrievalResult). The RetrievalResult
            reflects the documents used for generation (post-grading/rerank).
        """
        # Default image inclusion to the config master switch when not specified
        if include_images is None:
            include_images = settings.enable_images

        # Prepare initial state
        initial_state: RAGState = {
            "question": question,
            "metadata_filter": metadata_filter,
            "include_images": include_images,
            "skip_grading": skip_grading,
            "retrieval_result": None,
            "relevant_text_docs": None,
            "relevant_table_docs": None,
            "generation": None,
            "error": None,
        }

        # Execute the graph. Attach Langfuse callbacks when enabled so the full
        # trace (retrieve → rerank → prompt → LLM) is captured in the dashboard.
        from app.core.observability import get_callbacks
        final_state = self.graph.invoke(
            initial_state,
            config={"callbacks": get_callbacks()},
        )

        # Check for errors
        if final_state.get("error"):
            logger.warning(f"Workflow completed with error: {final_state['error']}")

        answer = final_state.get("generation", "No answer generated.")
        retrieval_result = final_state.get("retrieval_result")

        return answer, retrieval_result
