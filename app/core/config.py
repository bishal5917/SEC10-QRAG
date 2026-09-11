"""
Configuration Module - Centralized application settings.

Uses Pydantic Settings for type-safe configuration with .env file support.
All paths, model names, and hyperparameters are defined here as a single
source of truth for the entire application.

Importantly, this module redirects ALL model downloads (HuggingFace, torch hub)
to a local `models_cache/` directory within the project. This keeps everything
self-contained — nothing leaks to ~/.cache or other system paths.
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ─── Load .env early ──────────────────────────────────────────────────────────
# Load the .env file before configuring model libraries below, so values like
# HF_TOKEN are available in the environment when those libraries initialize.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# ─── Redirect model caches to local project directory ─────────────────────────
# This MUST run before any model library imports (torch, transformers, etc.)
# so that when they initialize, they see the overridden cache paths.
_MODELS_CACHE = _PROJECT_ROOT / "models_cache"
_MODELS_CACHE.mkdir(parents=True, exist_ok=True)

# HuggingFace hub (transformers, sentence-transformers)
os.environ["HF_HOME"] = str(_MODELS_CACHE / "huggingface")
# Torch hub (OpenCLIP weights)
os.environ["TORCH_HOME"] = str(_MODELS_CACHE / "torch")
# XDG cache fallback (some libraries use this)
os.environ["XDG_CACHE_HOME"] = str(_MODELS_CACHE)

# HuggingFace authentication token (removes anonymous-download warnings and
# raises rate limits). load_dotenv above already put HF_TOKEN in the environment;
# mirror it to HUGGING_FACE_HUB_TOKEN, which some HF libraries look for instead.
_hf_token = os.getenv("HF_TOKEN")
if _hf_token:
    os.environ["HUGGING_FACE_HUB_TOKEN"] = _hf_token

# ─── Fix SSL certificate verification on macOS ───────────────────────────────
# Some macOS Python installations lack root CA certificates, causing SSL errors
# when downloading pretrained weights from HuggingFace Hub and PyTorch Hub.
# This disables SSL verification for urllib — safe for model downloads.
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables and .env file.

    Hierarchy (highest precedence first):
        1. Environment variables
        2. .env file
        3. Default values defined below
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ─── Paths ────────────────────────────────────────────────────────────────
    project_root: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent.parent,
        description="Root directory of the project",
    )

    @property
    def data_dir(self) -> Path:
        """Directory containing source data files."""
        return self.project_root / "data"

    @property
    def pdf_dir(self) -> Path:
        """Directory containing PDF files to ingest."""
        return self.data_dir / "pdfs"

    @property
    def extracted_dir(self) -> Path:
        """Directory for extracted page images and intermediate outputs."""
        path = self.project_root / "extracted"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def images_dir(self) -> Path:
        """Directory for rendered page images."""
        path = self.extracted_dir / "page_images"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def vectorstore_dir(self) -> Path:
        """Directory for Qdrant persistent storage (embedded mode)."""
        path = self.project_root / "vector_store"
        path.mkdir(parents=True, exist_ok=True)
        return path

    # ─── API Keys ─────────────────────────────────────────────────────────────
    # Keeps Field() for the env-var alias mapping (GEMINI_API_KEY → gemini_api_key)
    gemini_api_key: Optional[str] = Field(default=None, alias="GEMINI_API_KEY")


    # ─── Logging & Observability ──────────────────────────────────────────────
    log_level: str = "INFO"                       # DEBUG | INFO | WARNING | ERROR
    show_retrieval_trace: bool = True             # print retrieved chunks + what's sent to the LLM
    langchain_debug: bool = True                  # LangChain built-in console tracing (retrieval, rerank, LLM steps)

    enable_langfuse: bool = True                 # master switch for Langfuse tracing
    langfuse_host: str = "http://localhost:3000"  # local self-hosted Langfuse URL
    langfuse_host: str = "https://us.cloud.langfuse.com" # cloud hosted
    langfuse_public_key: Optional[str] = Field(default=None, alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: Optional[str] = Field(default=None, alias="LANGFUSE_SECRET_KEY")

    # ─── LLM (Gemini) ─────────────────────────────────────────────────────────

    """
    gemini-3.6-flash (20/D)
    gemini-3.5-flash-lite (500/D)
    gemini-3.1-flash-lite (500/D)
    gemma-4-31b-it (14.4k/D)
    gemma-4-26b-a4b-it (14.4k/D)
    """

    gemini: str = "gemini-3.1-flash-lite"
    gemma: str = "gemma-4-31b-it"

    gemini_model: str = gemini      
    generation_temperature: float = 0.2          # lower = more factual
    max_output_tokens: int = 4096                # max tokens per response

    # ─── Text Embeddings (BGE, for text + tables) ─────────────────────────────
    text_embedding_model: str = "BAAI/bge-large-en-v1.5"
    text_embedding_dimension: int = 1024

    # ─── Image Embeddings (CLIP) ──────────────────────────────────────────────
    enable_images: bool = False                  # master switch for the image path (retrieval + sending to LLM)
    clip_model_name: str = "ViT-B-32-quickgelu"  # quickgelu matches OpenAI weights
    clip_pretrained: str = "openai"
    clip_embedding_dimension: int = 512

    # ─── PDF Processing ───────────────────────────────────────────────────────
    chunk_size: int = 1000                        # characters per text chunk
    chunk_overlap: int = 200                      # overlap between chunks
    page_image_dpi: int = 200                     # DPI when rendering pages to images
    min_chunk_size: int = 120                     # merge chunks smaller than this (avoids lone-heading slivers)

    # Layout-aware section-heading patterns (regex). The splitter breaks at
    # these FIRST so 10-Q sections stay coherent (heading + its content together).
    section_heading_patterns: list[str] = Field(
        default_factory=lambda: [
            r"\nNote\s+\d+\s*[:.]",           # "Note 7:" / "Note 7." financial statement notes
            r"\nItem\s+\d+[A-Z]?\s*[.:]",     # "Item 2." MD&A / other items
            r"\nNOTE\s+\d+",                  # "NOTE 7" all-caps variant
            r"\nCONSOLIDATED\s+[A-Z ]+",      # statement titles (all-caps)
        ]
    )

    # ─── Collections ─────────────────────────────────────────────────
    text_collection: str = "text_chunks"
    table_collection: str = "table_chunks"
    image_collection: str = "image_chunks"

    # ─── Retrieval (candidates fetched before re-ranking) ─────────────────────
    top_k_text: int = 30                          # text chunks to retrieve
    top_k_tables: int = 15                         # table chunks to retrieve
    top_k_images: int = 5                         # page images to retrieve
    enable_auto_filter: bool = True               # auto-filter by company/quarter named in query
    query_optimization: str = "llm"               # "none" | "llm" — rewrite query before retrieval

    # ─── Re-ranking (local cross-encoder, no API calls) ───────────────────────
    enable_reranker: bool = True                  # improves retrieval quality, free
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_top_n: int = 10                         # text chunks kept after re-ranking
    rerank_top_n_tables: int = 10                  # table chunks kept after re-ranking

    # ─── Relevance Grading ────────────────────────────────────────────────────
    relevance_threshold: float = 0.7              # min cosine sim to count as relevant

    # ─── Evaluation (all controlled here, no CLI arguments) ───────────────────
    eval_dataset: str = "data/qna_data_mini.csv"  # Q&A CSV to evaluate against
    eval_limit: Optional[int] = None              # max questions (None = all)
    eval_use_llm_judge: bool = True               # LLM-as-judge (uses extra API calls)
    eval_include_traditional: bool = True         # BLEU/METEOR/ROUGE-L/BERTScore
    eval_output_path: str = "output/evaluation_results.csv"

    # Rate limiting
    eval_delay_before_judge: float = 15.0         # wait between generation and judge
    eval_delay_between_rows: float = 30.0         # wait between questions

    @field_validator("gemini_api_key")
    @classmethod
    def validate_api_key(cls, v: Optional[str]) -> Optional[str]:
        """Treat placeholder keys (e.g. 'your_...') as unset."""
        if v and v.startswith("your_"):
            return None
        return v


# ─── Singleton instance ──────────────────────────────────────────────────────
# Import `settings` from here across the application for consistent config access.
settings = Settings()
