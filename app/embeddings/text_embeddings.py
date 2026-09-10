"""
Text Embedding Module.

Implements a LangChain-compatible Embeddings interface using sentence-transformers
(BGE-large-en-v1.5) for text and table content embedding.

Design Choices:
    - BGE-large-en-v1.5 is one of the top-performing open-source embedding models
      on MTEB benchmarks, particularly strong for retrieval tasks.
    - Implements LangChain's Embeddings interface so it can be plugged directly
      into LangChain vector stores, retrievers, and chains.
    - Normalizes embeddings to unit vectors for cosine similarity compatibility.
    - Cross-platform GPU support: CUDA (Windows/Linux), MPS (macOS), CPU (fallback).
"""

from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.core.device import get_device
from app.core.logging import get_logger

logger = get_logger(__name__)


class BGETextEmbeddings(Embeddings):
    """
    LangChain-compatible embedding wrapper around sentence-transformers BGE model.

    This class implements the LangChain Embeddings interface, making it
    compatible with LangChain's Qdrant, FAISS, and other vector store integrations.

    Features:
        - High-quality 1024-dimensional embeddings
        - Normalized vectors for cosine similarity
        - Batch processing with configurable batch size
        - Cross-platform GPU acceleration (CUDA on Windows/Linux, MPS on macOS)

    Usage:
        embeddings = BGETextEmbeddings()
        vectors = embeddings.embed_documents(["Hello world", "Financial report"])
        query_vector = embeddings.embed_query("What is the revenue?")
    """

    def __init__(self, model_name: str = None, batch_size: int = 32):
        """
        Initialize the BGE text embedding model.

        Automatically selects the best available accelerator:
            - CUDA (NVIDIA GPU on Windows/Linux)
            - MPS (Apple Silicon on macOS)
            - CPU (universal fallback)

        Args:
            model_name: HuggingFace model identifier.
                        Defaults to settings.text_embedding_model.
            batch_size: Number of texts to encode simultaneously.
                        Higher values use more memory but are faster.
        """
        self.model_name = model_name or settings.text_embedding_model
        self.batch_size = batch_size

        # Cross-platform device selection: CUDA > MPS > CPU
        device = get_device()

        logger.info(f"Loading text embedding model: {self.model_name} (device={device})")
        self.model = SentenceTransformer(self.model_name, device=device)

        # Log model info
        self.dimension = self.model.get_embedding_dimension()
        logger.info(
            f"Text embeddings ready: dim={self.dimension}, "
            f"device={self.model.device}"
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of documents (texts/tables).

        This is used during ingestion to embed all document chunks.
        BGE models benefit from a query prefix for retrieval tasks,
        but for documents we embed as-is.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors (each is a list of floats).
        """
        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > 50,  # Show progress for large batches
            normalize_embeddings=True,
        )

        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query text.

        For BGE models, prepending "Represent this sentence:" improves
        retrieval performance. This method handles the query formatting.

        Args:
            text: The user's query string.

        Returns:
            Embedding vector as a list of floats.
        """
        # BGE models perform better with a retrieval instruction prefix
        formatted_query = f"Represent this sentence for searching relevant passages: {text}"

        embedding = self.model.encode(
            [formatted_query],
            normalize_embeddings=True,
        )

        return embedding[0].tolist()
