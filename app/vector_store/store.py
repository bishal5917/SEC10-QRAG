"""
Vector Store Module (Qdrant, embedded, hybrid search).

Manages Qdrant collections for multi-modal document storage using
LangChain's QdrantVectorStore integration. Runs entirely locally in
EMBEDDED mode (on-disk, in-process) — no Docker, no server.

Why Qdrant + hybrid:
    Qdrant natively stores BOTH dense and sparse vectors per point and fuses
    them server-side (RRF), giving true hybrid search:
        - Dense (BGE embeddings)   → semantic / meaning matches
        - Sparse (FastEmbed BM25)  → keyword / exact-term matches
    This is stronger than dense-only for financial docs full of specific
    numbers, tickers, and exact terminology.

Architecture:
    Three collections, each on the same embedded Qdrant client:
        1. text_chunks   — HYBRID (dense BGE + sparse) text passages
        2. table_chunks  — HYBRID (dense BGE + sparse) Markdown tables
        3. image_chunks  — DENSE only (CLIP); image path is optional/disabled

Storage:
    QdrantClient(path=...) persists to ./vector_store/ on disk. Data survives
    restarts. Re-ingestion only needed when documents change.
"""

from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Sparse embedding model (BM25-style, runs locally via fastembed). Shared across
# the text and table collections. Qdrant uses this for the keyword side of hybrid.
_SPARSE_MODEL = "Qdrant/bm25"


class MultiModalVectorStore:
    """
    Multi-collection Qdrant vector store (embedded, hybrid for text/tables).

    Text and table collections use HYBRID retrieval (dense BGE + sparse BM25);
    the image collection is dense-only (CLIP). All collections live on a single
    embedded Qdrant client persisted to disk.

    Usage:
        store = MultiModalVectorStore(
            text_embeddings=BGETextEmbeddings(),
            image_embeddings=CLIPImageEmbeddings(),
        )
        store.add_text_documents(text_chunks)
        results = store.text_store.similarity_search("revenue growth", k=5)
    """

    def __init__(
        self,
        text_embeddings: Embeddings,
        image_embeddings: Embeddings,
        persist_dir: Path = None,
    ):
        """
        Initialize the embedded Qdrant client and the three collections.

        Args:
            text_embeddings: Dense embeddings for text/tables (BGE).
            image_embeddings: Dense embeddings for images (CLIP).
            persist_dir: On-disk storage dir. Defaults to settings.vectorstore_dir.
        """
        self.persist_dir = str(persist_dir or settings.vectorstore_dir)

        # Embedded Qdrant client (local, on-disk, no server/Docker)
        self._client = QdrantClient(path=self.persist_dir)

        # Sparse embeddings for the hybrid keyword side (local BM25 via fastembed)
        self._sparse = FastEmbedSparse(model_name=_SPARSE_MODEL)

        self._text_embeddings = text_embeddings
        self._image_embeddings = image_embeddings

        # Ensure collections exist with the right vector configuration
        self._ensure_collection(settings.text_collection, dense_dim=settings.text_embedding_dimension, hybrid=True)
        self._ensure_collection(settings.table_collection, dense_dim=settings.text_embedding_dimension, hybrid=True)
        self._ensure_collection(settings.image_collection, dense_dim=settings.clip_embedding_dimension, hybrid=False)

        # LangChain vector store wrappers (these emit callback events when used
        # via .as_retriever(), so retrieval shows up in LangChain tracing).
        self.text_store = self._build_store(settings.text_collection, text_embeddings, hybrid=True)
        self.table_store = self._build_store(settings.table_collection, text_embeddings, hybrid=True)
        self.image_store = self._build_store(settings.image_collection, image_embeddings, hybrid=False)

        stats = self.get_stats()
        logger.info(
            f"Qdrant vector store (embedded) at {self.persist_dir} | "
            f"Text: {stats['text']}, Tables: {stats['tables']}, Images: {stats['images']}"
        )

    # ─── Collection setup ─────────────────────────────────────────────────────

    def _ensure_collection(self, name: str, dense_dim: int, hybrid: bool) -> None:
        """
        Create a Qdrant collection if it doesn't already exist.

        Named vectors are used so LangChain's QdrantVectorStore can address the
        dense vector as "dense" and (for hybrid) the sparse vector as "sparse".

        Args:
            name: Collection name.
            dense_dim: Dimension of the dense (BGE/CLIP) vectors.
            hybrid: Whether to also configure a sparse vector (text/tables).
        """
        if self._client.collection_exists(name):
            return

        vectors_config = {
            "dense": models.VectorParams(size=dense_dim, distance=models.Distance.COSINE)
        }
        sparse_config = (
            {"sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)}
            if hybrid else None
        )

        self._client.create_collection(
            collection_name=name,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_config,
        )
        logger.info(f"Created Qdrant collection '{name}' (hybrid={hybrid}, dim={dense_dim})")

    def _build_store(self, name: str, dense_embeddings: Embeddings, hybrid: bool) -> QdrantVectorStore:
        """
        Build a LangChain QdrantVectorStore over an existing collection.

        Args:
            name: Collection name.
            dense_embeddings: Dense embedding model for this collection.
            hybrid: If True, use HYBRID retrieval (dense + sparse); else DENSE.

        Returns:
            Configured QdrantVectorStore.
        """
        kwargs = dict(
            client=self._client,
            collection_name=name,
            embedding=dense_embeddings,
            vector_name="dense",
        )
        if hybrid:
            kwargs.update(
                retrieval_mode=RetrievalMode.HYBRID,
                sparse_embedding=self._sparse,
                sparse_vector_name="sparse",
            )
        else:
            kwargs.update(retrieval_mode=RetrievalMode.DENSE)

        return QdrantVectorStore(**kwargs)

    # ─── Ingestion ────────────────────────────────────────────────────────────

    def add_text_documents(self, documents: list[Document]) -> None:
        """Add text chunk documents (dense + sparse vectors computed automatically)."""
        if not documents:
            logger.warning("No text documents to add")
            return
        self.text_store.add_documents(documents)
        logger.info(f"Added {len(documents)} text chunks to Qdrant")

    def add_table_documents(self, documents: list[Document]) -> None:
        """Add table (Markdown) documents (dense + sparse vectors computed automatically)."""
        if not documents:
            logger.warning("No table documents to add")
            return
        self.table_store.add_documents(documents)
        logger.info(f"Added {len(documents)} table chunks to Qdrant")

    def add_image_documents(
        self,
        documents: list[Document],
        image_embeddings: list[list[float]],
    ) -> None:
        """
        Add image documents with pre-computed CLIP embeddings (dense only).

        Args:
            documents: LangChain Documents with image metadata.
            image_embeddings: Pre-computed CLIP dense vectors for the images.
        """
        if not documents:
            logger.warning("No image documents to add")
            return

        points = []
        for i, (doc, vec) in enumerate(zip(documents, image_embeddings)):
            points.append(
                models.PointStruct(
                    id=i,
                    vector={"dense": vec},
                    payload={
                        "page_content": doc.page_content,
                        "metadata": doc.metadata,
                    },
                )
            )
        self._client.upsert(collection_name=settings.image_collection, points=points)
        logger.info(f"Added {len(documents)} image embeddings to Qdrant")

    # ─── Stats & maintenance ──────────────────────────────────────────────────

    def _count(self, name: str) -> int:
        """Return the number of points in a collection (0 if missing)."""
        try:
            return self._client.count(collection_name=name, exact=True).count
        except Exception:
            return 0

    def get_stats(self) -> dict[str, int]:
        """Get item counts for each collection."""
        return {
            "text": self._count(settings.text_collection),
            "tables": self._count(settings.table_collection),
            "images": self._count(settings.image_collection),
        }

    def clear_all(self) -> None:
        """
        Delete and recreate all collections on the SAME client (destructive).

        Recreating on the existing client avoids Qdrant's embedded-mode
        single-client lock (a second QdrantClient on the same folder would
        raise "already accessed by another instance").
        """
        for name in (settings.text_collection, settings.table_collection, settings.image_collection):
            if self._client.collection_exists(name):
                self._client.delete_collection(name)
        logger.warning("All Qdrant collections cleared; recreating empty collections")

        # Recreate empty collections and rebuild the LangChain store wrappers
        self._ensure_collection(settings.text_collection, dense_dim=settings.text_embedding_dimension, hybrid=True)
        self._ensure_collection(settings.table_collection, dense_dim=settings.text_embedding_dimension, hybrid=True)
        self._ensure_collection(settings.image_collection, dense_dim=settings.clip_embedding_dimension, hybrid=False)

        self.text_store = self._build_store(settings.text_collection, self._text_embeddings, hybrid=True)
        self.table_store = self._build_store(settings.table_collection, self._text_embeddings, hybrid=True)
        self.image_store = self._build_store(settings.image_collection, self._image_embeddings, hybrid=False)

    def close(self) -> None:
        """
        Explicitly close the embedded Qdrant client.

        Closing here (while the interpreter is healthy) avoids the noisy
        teardown errors that occur when QdrantClient.__del__ runs during
        Python shutdown (when imports/meta_path are already gone).
        Safe to call multiple times.
        """
        try:
            self._client.close()
        except Exception:
            pass
