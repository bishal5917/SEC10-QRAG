"""
Multi-Modal Retriever Module.

Orchestrates retrieval across text, table, and image collections,
combining results into a unified context for the generation step.

Architecture:
    The retriever performs parallel searches across three collections:
        1. Text collection (BGE embeddings) → relevant text passages
        2. Table collection (BGE embeddings) → relevant financial tables
        3. Image collection (CLIP embeddings) → relevant page images

    Query embedding strategy:
        - Text/Tables: Query is embedded with BGE (same space as documents)
        - Images: Query is embedded with CLIP text encoder (cross-modal retrieval)

    Results are scored by cosine distance and combined into a RetrievalResult
    data class for consumption by the generation chain.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document

from app.core.config import settings
from app.core.logging import get_logger
from app.embeddings.clip_embeddings import CLIPImageEmbeddings
from app.embeddings.text_embeddings import BGETextEmbeddings
from app.vectorstore.store import MultiModalVectorStore

logger = get_logger(__name__)


@dataclass
class RetrievalResult:
    """
    Unified container for multi-modal retrieval results.

    Holds text passages, tables, and image paths retrieved from the
    vector store, along with helper methods to format context for
    the LLM generation step.

    Attributes:
        text_documents: Retrieved text passage Documents.
        table_documents: Retrieved table Documents (Markdown format).
        image_documents: Retrieved image Documents (with image_path in metadata).
        query: The original user query.
    """

    query: str = ""                    # original user question (used for the answer)
    search_query: str = ""             # retrieval-optimized query (used for retrieval + reranking)
    text_documents: list[Document] = field(default_factory=list)
    table_documents: list[Document] = field(default_factory=list)
    image_documents: list[Document] = field(default_factory=list)

    @property
    def has_results(self) -> bool:
        """Check if any results were retrieved from any modality."""
        return bool(
            self.text_documents or self.table_documents or self.image_documents
        )

    def get_text_context(self) -> str:
        """
        Format text and table results into a structured context string.

        This string is injected into the LLM prompt as retrieved context.
        Tables are clearly delimited to help the LLM parse them.

        Returns:
            Formatted context string combining text passages and tables.
        """
        context_parts = []

        # Format text passages
        if self.text_documents:
            context_parts.append("══════ RELEVANT TEXT PASSAGES ══════")
            for i, doc in enumerate(self.text_documents, 1):
                source = doc.metadata.get("filename", "Unknown")
                page = doc.metadata.get("page_number", "?")
                ticker = doc.metadata.get("ticker", "")
                context_parts.append(
                    f"\n📄 [Passage {i}] {ticker} — {source}, Page {page}\n"
                    f"{doc.page_content}"
                )

        # Format tables (Markdown preserved for LLM readability)
        if self.table_documents:
            context_parts.append("\n\n══════ RELEVANT TABLES ══════")
            for i, doc in enumerate(self.table_documents, 1):
                source = doc.metadata.get("filename", "Unknown")
                page = doc.metadata.get("page_number", "?")
                ticker = doc.metadata.get("ticker", "")
                context_parts.append(
                    f"\n📊 [Table {i}] {ticker} — {source}, Page {page}\n"
                    f"{doc.page_content}"
                )

        return "\n".join(context_parts)

    def get_image_paths(self) -> list[str]:
        """
        Get file paths of retrieved page images.

        Only returns paths that actually exist on disk (filters stale entries).

        Returns:
            List of valid image file paths.
        """
        paths = []
        for doc in self.image_documents:
            path = doc.metadata.get("image_path", "")
            if path and Path(path).exists():
                paths.append(path)
        return paths

    def get_source_summary(self) -> str:
        """
        Get a brief summary of retrieval sources for logging/display.

        Returns:
            Human-readable summary of what was retrieved.
        """
        return (
            f"Retrieved: {len(self.text_documents)} text passages, "
            f"{len(self.table_documents)} tables, "
            f"{len(self.image_documents)} page images"
        )

    def log_trace(self) -> None:
        """
        Log a detailed, human-readable trace of everything retrieved.

        Shows, for each retrieved document: source PDF, page number, and a
        short snippet of the content. Useful for debugging retrieval quality
        and understanding exactly what context the LLM will receive.

        Only emits when settings.show_retrieval_trace is True.
        """
        from app.core.config import settings

        if not settings.show_retrieval_trace:
            return

        def _fmt(doc, idx):
            meta = doc.metadata
            ticker = meta.get("ticker", "?")
            year = meta.get("year", "?")
            quarter = meta.get("quarter", "?")
            page = meta.get("page_number", "?")
            snippet = (doc.page_content or "").replace("\n", " ").strip()[:100]
            return f"    [{idx}] {ticker} {year} {quarter}, p.{page}  \"{snippet}...\""

        logger.info(f"📥 RETRIEVED for query: \"{self.query[:70]}\"")

        logger.info(f"  TEXT ({len(self.text_documents)}):")
        for i, doc in enumerate(self.text_documents, 1):
            logger.info(_fmt(doc, i))

        logger.info(f"  TABLES ({len(self.table_documents)}):")
        for i, doc in enumerate(self.table_documents, 1):
            logger.info(_fmt(doc, i))

        logger.info(f"  IMAGES ({len(self.image_documents)}):")
        for i, doc in enumerate(self.image_documents, 1):
            logger.info(_fmt(doc, i))


class MultiModalRetriever:
    """
    Retriever that searches across text, table, and image vector collections.

    Combines LangChain's similarity_search with CLIP cross-modal retrieval
    to find relevant context from all modalities for a user query.

    Usage:
        retriever = MultiModalRetriever(vector_store, text_embedder, clip_embedder)
        result = retriever.retrieve("What was Apple's revenue in Q3 2023?")
        # result.text_documents → relevant passages
        # result.table_documents → relevant tables
        # result.get_image_paths() → paths to relevant page images
    """

    def __init__(
        self,
        vector_store: MultiModalVectorStore,
        text_embeddings: BGETextEmbeddings,
        clip_embeddings: CLIPImageEmbeddings,
    ):
        """
        Initialize the multi-modal retriever.

        Args:
            vector_store: MultiModalVectorStore with all collections loaded.
            text_embeddings: BGE embeddings for text/table query encoding.
            clip_embeddings: CLIP embeddings for image query encoding.
        """
        self.vector_store = vector_store
        self.text_embeddings = text_embeddings
        self.clip_embeddings = clip_embeddings

    def _multi_query_search(
        self,
        store,
        queries: list[str],
        k: int,
        qdrant_filter=None,
    ) -> list[Document]:
        """
        Hybrid-search a Qdrant store with multiple queries and merge results.

        Each query runs through the store's .as_retriever().invoke() (traced,
        hybrid dense+sparse). Results are merged and de-duplicated so the same
        underlying chunk isn't repeated. This keeps exact-number chunks (found
        by the original query's sparse side) in the pool alongside the rewrite's
        broader matches, before the reranker sorts everything.

        De-dup key: (source, page_number, chunk/table index).

        Args:
            store: A LangChain QdrantVectorStore (text_store or table_store).
            queries: Query strings to search with (original + rewritten).
            k: Results to fetch per query.
            qdrant_filter: Optional Qdrant Filter for metadata narrowing.

        Returns:
            Merged, de-duplicated list of Documents (first query's hits first).
        """
        search_kwargs = {"k": k}
        if qdrant_filter is not None:
            search_kwargs["filter"] = qdrant_filter

        retriever = store.as_retriever(search_kwargs=search_kwargs)

        merged: list[Document] = []
        seen: set = set()

        for q in queries:
            for doc in retriever.invoke(q):
                meta = doc.metadata
                key = (
                    meta.get("source"),
                    meta.get("page_number"),
                    meta.get("chunk_index", meta.get("table_index")),
                )
                if key not in seen:
                    seen.add(key)
                    merged.append(doc)

        return merged

    def retrieve(
        self,
        query: str,
        top_k_text: int = None,
        top_k_tables: int = None,
        top_k_images: int = None,
        metadata_filter: Optional[dict] = None,
    ) -> RetrievalResult:
        """
        Perform multi-modal retrieval for a user query.

        Searches all three collections in parallel (text, tables, images)
        and combines results into a RetrievalResult.

        Args:
            query: The user's natural language question.
            top_k_text: Number of text passages to retrieve (default from settings).
            top_k_tables: Number of tables to retrieve (default from settings).
            top_k_images: Number of images to retrieve (default from settings).
            metadata_filter: Optional neutral filter dict (e.g. {"tickers": ["AAPL"]})
                            translated to a Qdrant filter for the search.

        Returns:
            RetrievalResult containing documents from all modalities.
        """
        top_k_text = top_k_text or settings.top_k_text
        top_k_tables = top_k_tables or settings.top_k_tables
        top_k_images = top_k_images or settings.top_k_images

        # Auto-derive a metadata filter from the ORIGINAL query (company/quarter).
        # Detection works best on the user's wording, so we filter before rewriting.
        # build_metadata_filter returns a DB-neutral dict; translate to a Qdrant
        # Filter for the actual search.
        from app.retrieval.query_filter import build_metadata_filter, to_qdrant_filter
        if metadata_filter is None and settings.enable_auto_filter:
            metadata_filter = build_metadata_filter(query)
        qdrant_filter = to_qdrant_filter(metadata_filter)

        # Optimize the query for retrieval (LLM rewrite to match document
        # vocabulary). We search with BOTH the ORIGINAL and REWRITTEN queries
        # (multi-query hybrid): the original's sparse/BM25 side reliably surfaces
        # exact-number chunks (e.g. "696.2%"), while the rewrite adds vocabulary
        # coverage. Merging both guarantees number-bearing chunks aren't missing;
        # the reranker then sorts the combined pool. The ORIGINAL query is still
        # what the answer is generated for.
        from app.retrieval.query_optimizer import optimize_query
        rewritten_query = optimize_query(query)

        search_queries = [query]
        if rewritten_query and rewritten_query.strip() != query.strip():
            search_queries.append(rewritten_query)

        # RetrievalResult keeps the ORIGINAL query (for the answer) and the
        # rewritten search_query (used for reranking, which aligns the rerank
        # scoring with how retrieval was actually performed).
        result = RetrievalResult(query=query, search_query=rewritten_query)

        # ─── Text Retrieval (HYBRID + multi-query merge) ──────────────────
        try:
            result.text_documents = self._multi_query_search(
                store=self.vector_store.text_store,
                queries=search_queries,
                k=top_k_text,
                qdrant_filter=qdrant_filter,
            )
        except Exception as e:
            logger.error(f"Text retrieval failed: {e}")

        # ─── Table Retrieval (HYBRID + multi-query merge) ─────────────────
        try:
            result.table_documents = self._multi_query_search(
                store=self.vector_store.table_store,
                queries=search_queries,
                k=top_k_tables,
                qdrant_filter=qdrant_filter,
            )
        except Exception as e:
            logger.error(f"Table retrieval failed: {e}")

        # ─── Image Retrieval (CLIP cross-modal, dense-only) ──────────────
        # Skip the entire image path when disabled in config (no CLIP query,
        # no image documents). Controlled by settings.enable_images.
        if settings.enable_images:
            try:
                # Embed the query with CLIP's text encoder, then search the
                # dense image collection directly via the Qdrant client.
                query_embedding = self.clip_embeddings.embed_query(rewritten_query)

                hits = self.vector_store._client.search(
                    collection_name=settings.image_collection,
                    query_vector=("dense", query_embedding),
                    limit=top_k_images,
                    query_filter=qdrant_filter,
                    with_payload=True,
                )

                for hit in hits:
                    payload = hit.payload or {}
                    result.image_documents.append(
                        Document(
                            page_content=payload.get("page_content", ""),
                            metadata=payload.get("metadata", {}),
                        )
                    )

            except Exception as e:
                logger.error(f"Image retrieval failed: {e}")

        logger.info(result.get_source_summary())
        result.log_trace()
        return result
