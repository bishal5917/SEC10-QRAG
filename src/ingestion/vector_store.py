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


def _embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Embed texts one-by-one to guarantee len(embeddings) == len(texts).
    If any single embedding fails, returns a zero vector as placeholder
    so the batch length never mismatches.
    """
    import httpx
    embeddings = []
    _dim = None  # Will be set from first successful embedding

    for i, text in enumerate(texts):
        try:
            # Replace problematic characters and ensure non-empty
            clean_text = text.strip()
            if not clean_text or len(clean_text) < 3:
                clean_text = "empty"

            resp = httpx.post(
                f"{OLLAMA_BASE_URL}/api/embeddings",
                json={"model": EMBED_MODEL, "prompt": clean_text},
                timeout=60.0,
            )
            resp.raise_for_status()
            embedding = resp.json()["embedding"]

            if not embedding:
                raise ValueError("Ollama returned empty embedding")

            if _dim is None:
                _dim = len(embedding)

            embeddings.append(embedding)
        except Exception as e:
            log.warning(f"Embedding failed for chunk {i} (len={len(text)}): {e}")
            # Use zero vector as fallback (will have low similarity to everything)
            if _dim is None:
                _dim = 768  # nomic-embed-text default dimension
            embeddings.append([0.0] * _dim)

    return embeddings


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

    try:
        client.delete_collection(COLLECTION_NAME)
        log.info(f"Dropped existing collection: {COLLECTION_NAME}")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    log.info(f"Created collection: {COLLECTION_NAME}")

    batch_size = 50
    t0 = time.perf_counter()

    # Drop chunks with empty text — embedding function returns fewer results, causing IndexError
    before = len(chunks)
    chunks = [c for c in chunks if c.get("text", "").strip()]
    if len(chunks) < before:
        log.warning(f"Dropped {before - len(chunks)} chunks with empty text before embedding")

    # Store image_b64 separately (too large for ChromaDB metadata)
    # Save to disk and reference by ID
    image_store_dir = CHROMA_DIR / "images"
    image_store_dir.mkdir(parents=True, exist_ok=True)

    total_batches = -(-len(chunks) // batch_size)
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i: i + batch_size]
        batch_num = i // batch_size + 1
        log.info(f"Embedding batch {batch_num}/{total_batches} ({len(batch)} chunks)...")
        t_batch = time.perf_counter()
        texts = [c["text"] for c in batch]
        embeddings = _embed_texts(texts)

        ids = [_make_id(c["source"], c["page"], i + j) for j, c in enumerate(batch)]

        # Save images to disk, store only a flag in metadata
        metadatas = []
        for j, c in enumerate(batch):
            meta = {
                "source": c["source"],
                "page": c["page"],
                "chunk_type": c["chunk_type"],
                "has_image": "true" if c.get("image_b64") else "false",
            }
            if c.get("image_b64"):
                img_path = image_store_dir / f"{ids[j]}.b64"
                img_path.write_text(c["image_b64"])
            metadatas.append(meta)

        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
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
