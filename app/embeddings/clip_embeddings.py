"""
CLIP Image Embedding Module.

Uses OpenCLIP (ViT-B-32) for embedding page images and for text-to-image
retrieval via CLIP's shared embedding space.

Architecture:
    CLIP produces embeddings in a shared vector space where:
        - Images and their textual descriptions are close together
        - A text query like "revenue bar chart" will be near images of revenue charts
    
    This enables cross-modal retrieval: embed the query with CLIP's text encoder
    and search against image embeddings to find visually relevant pages.

Design Choices:
    - ViT-B-32 provides a good balance of speed and quality (512-dim embeddings).
    - Cross-platform GPU support: CUDA (Windows/Linux), MPS (macOS), CPU (fallback).
    - Implements LangChain's Embeddings interface for vector store compatibility.
    - Separate methods for image embedding (from files) and text embedding (for queries).
"""

from pathlib import Path

import numpy as np
import open_clip
import torch
from langchain_core.embeddings import Embeddings
from PIL import Image

from app.core.config import settings
from app.core.device import get_device
from app.core.logging import get_logger

logger = get_logger(__name__)


class CLIPImageEmbeddings(Embeddings):
    """
    LangChain-compatible embedding class using OpenCLIP for images.

    Provides two key capabilities:
        1. embed_images(): Encode image files into CLIP vector space
        2. embed_query(): Encode text into CLIP vector space (for text→image retrieval)

    The embed_documents() method is also available for compatibility with
    LangChain vector stores (embeds text descriptions of images).

    Usage:
        clip = CLIPImageEmbeddings()

        # Embed images from file paths
        image_vectors = clip.embed_images(["/path/to/page1.png", "/path/to/page2.png"])

        # Embed a text query for image retrieval
        query_vector = clip.embed_query("revenue growth chart")
    """

    def __init__(
        self,
        model_name: str = None,
        pretrained: str = None,
    ):
        """
        Initialize the OpenCLIP model.

        Loads the CLIP model and its image preprocessing pipeline.
        Automatically selects the best available accelerator:
            - CUDA (NVIDIA GPU on Windows/Linux)
            - MPS (Apple Silicon on macOS)
            - CPU (universal fallback)

        Args:
            model_name: CLIP architecture name (default: ViT-B-32).
            pretrained: Pretrained weights source (default: openai).
        """
        self.model_name = model_name or settings.clip_model_name
        self.pretrained = pretrained or settings.clip_pretrained

        # Cross-platform device selection: CUDA > MPS > CPU
        self.device = get_device()

        logger.info(
            f"Loading CLIP model: {self.model_name} "
            f"(pretrained={self.pretrained}, device={self.device})"
        )

        # Load model, preprocessing transforms, and tokenizer
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained
        )
        self.model = self.model.to(self.device)
        self.model.eval()  # Set to inference mode (no dropout, etc.)

        self.tokenizer = open_clip.get_tokenizer(self.model_name)
        self.dimension = self.model.visual.output_dim

        logger.info(f"CLIP embeddings ready: dim={self.dimension}, device={self.device}")

    def embed_images(self, image_paths: list[str], batch_size: int = 8) -> list[list[float]]:
        """
        Embed images from file paths using CLIP's visual encoder.

        Processes images in batches for memory efficiency.
        Each image is preprocessed (resized, normalized) before encoding.

        Args:
            image_paths: List of file paths to PNG/JPG images.
            batch_size: Number of images to process simultaneously.

        Returns:
            List of normalized embedding vectors (each is a list of floats).
        """
        if not image_paths:
            return []

        all_embeddings = []

        for batch_start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[batch_start : batch_start + batch_size]
            batch_tensors = []

            for img_path in batch_paths:
                try:
                    # Load and preprocess the image
                    image = Image.open(img_path).convert("RGB")
                    tensor = self.preprocess(image)
                    batch_tensors.append(tensor)
                except Exception as e:
                    logger.warning(f"Failed to load image {img_path}: {e}")
                    # Use zero tensor as fallback (will produce near-zero embedding)
                    batch_tensors.append(torch.zeros(3, 224, 224))

            # Stack into batch and encode
            batch = torch.stack(batch_tensors).to(self.device)

            with torch.no_grad():
                embeddings = self.model.encode_image(batch)
                # L2-normalize for cosine similarity
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

            all_embeddings.extend(embeddings.cpu().numpy().tolist())

        return all_embeddings

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed text descriptions using CLIP's text encoder.

        This satisfies the LangChain Embeddings interface. Used when
        storing image metadata as documents in a vector store.

        Note: For image retrieval, use embed_images() with actual image files.
              This method is for text-based fallback or hybrid approaches.

        Args:
            texts: List of text descriptions to embed.

        Returns:
            List of CLIP text embedding vectors.
        """
        if not texts:
            return []

        all_embeddings = []
        batch_size = 32

        for batch_start in range(0, len(texts), batch_size):
            batch_texts = texts[batch_start : batch_start + batch_size]
            tokens = self.tokenizer(batch_texts).to(self.device)

            with torch.no_grad():
                text_embeddings = self.model.encode_text(tokens)
                text_embeddings = text_embeddings / text_embeddings.norm(
                    dim=-1, keepdim=True
                )

            all_embeddings.extend(text_embeddings.cpu().numpy().tolist())

        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        """
        Embed a text query for cross-modal image retrieval.

        Uses CLIP's text encoder to project the query into the same
        vector space as images. This enables finding visually relevant
        pages using natural language queries.

        Example: "revenue breakdown pie chart" → finds pages with pie charts

        Args:
            text: Query text describing the visual content to find.

        Returns:
            CLIP text embedding vector (same space as image embeddings).
        """
        tokens = self.tokenizer([text]).to(self.device)

        with torch.no_grad():
            embedding = self.model.encode_text(tokens)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)

        return embedding.cpu().numpy()[0].tolist()
