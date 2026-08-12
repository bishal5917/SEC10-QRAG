import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data" / "pdfs"))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", BASE_DIR / "data" / "chroma"))

# Ollama
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3:instruct")
VISION_MODEL = os.getenv("VISION_MODEL", "llava")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

# Chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))

# Retrieval
TOP_K = int(os.getenv("TOP_K", 8))
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "true").lower() in ("true", "1", "yes")
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", 20))  # Over-retrieve before reranking
COLLECTION_NAME = "sec_10q_docs"

# Figure/Chart Detection (disabled by default — text extraction captures chart data)
FIGURE_DETECTION_ENABLED = os.getenv("FIGURE_DETECTION_ENABLED", "false").lower() in ("true", "1", "yes")
