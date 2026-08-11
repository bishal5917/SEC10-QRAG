import re
import time
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    CHROMA_DIR, COLLECTION_NAME, CHUNK_SIZE, CHUNK_OVERLAP,
    OLLAMA_BASE_URL, EMBED_MODEL,
)
from src.logger import get_logger

log = get_logger("vector_store")


def _get_chroma_client() -> chromadb.ClientAPI:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    log.debug(f"ChromaDB path: {CHROMA_DIR}")
    return chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )


def _get_embedding_function():
    from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
    log.debug(f"Embedding model: {EMBED_MODEL} via {OLLAMA_BASE_URL}")
    return OllamaEmbeddingFunction(
        url=f"{OLLAMA_BASE_URL}/api/embeddings",
        model_name=EMBED_MODEL,
    )


def _split_text_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )
    result = []
    for chunk in chunks:
        if chunk["chunk_type"] in ("table", "figure", "image"):
            result.append(chunk)
        else:
            sub_texts = splitter.split_text(chunk["text"])
            for sub in sub_texts:
                result.append({**chunk, "text": sub})
    return result


def _make_id(source: str, page: int, idx: int) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", source)
    return f"{safe}__p{page}__i{idx}"


def build_vector_store(raw_chunks: List[Dict[str, Any]]) -> chromadb.Collection:
    log.info(f"Splitting {len(raw_chunks)} raw chunks...")
    chunks = _split_text_chunks(raw_chunks)

    by_type: dict = {}
    for c in chunks:
        by_type[c["chunk_type"]] = by_type.get(c["chunk_type"], 0) + 1
    log.info(f"Total chunks after splitting: {len(chunks)} | breakdown: {by_type}")

    client = _get_chroma_client()
    ef = _get_embedding_function()

    try:
        client.delete_collection(COLLECTION_NAME)
        log.info(f"Dropped existing collection: {COLLECTION_NAME}")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    log.info(f"Created collection: {COLLECTION_NAME}")

    batch_size = 50
    total_batches = -(-len(chunks) // batch_size)
    t0 = time.perf_counter()

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i: i + batch_size]
        batch_num = i // batch_size + 1
        log.info(f"Embedding batch {batch_num}/{total_batches} ({len(batch)} chunks)...")
        t_batch = time.perf_counter()
        collection.add(
            ids=[_make_id(c["source"], c["page"], i + j) for j, c in enumerate(batch)],
            documents=[c["text"] for c in batch],
            metadatas=[
                {
                    "source": c["source"],
                    "page": c["page"],
                    "chunk_type": c["chunk_type"],
                    # Store image_b64 only for figure chunks — empty string otherwise
                    # (ChromaDB metadata values must be str/int/float/bool)
                    "image_b64": c.get("image_b64", ""),
                }
                for c in batch
            ],
        )
        log.info(f"Batch {batch_num}/{total_batches} done in {(time.perf_counter()-t_batch)*1000:.1f}ms")

    total_elapsed = time.perf_counter() - t0
    log.info(f"Vector store built: {collection.count()} vectors in {total_elapsed:.1f}s")
    return collection


def get_collection() -> chromadb.Collection:
    client = _get_chroma_client()
    ef = _get_embedding_function()
    log.debug(f"Loading collection: {COLLECTION_NAME}")
    return client.get_collection(name=COLLECTION_NAME, embedding_function=ef)
