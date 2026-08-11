import time
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional

from src.pipeline import RAGPipeline
from src.logger import get_logger

log = get_logger("app")
STATIC_DIR = Path(__file__).parent / "static"

pipeline: Optional[RAGPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # global pipeline
    log.info("Starting up — initializing RAG pipeline...")
    # pipeline = RAGPipeline()
    # log.info("RAG pipeline ready")
    yield
    log.info("Shutting down")


app = FastAPI(
    title="SEC 10-Q RAG API",
    description="RAG-based information retrieval over SEC 10-Q filings",
    version="1.0.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    log.info(f"→ {request.method} {request.url.path}")
    response = await call_next(request)
    elapsed = (time.perf_counter() - start) * 1000
    log.info(f"← {request.method} {request.url.path} [{response.status_code}] {elapsed:.1f}ms")
    return response


@app.get("/", include_in_schema=False)
def ui():
    return FileResponse(STATIC_DIR / "index.html")


class QueryRequest(BaseModel):
    question: str
    top_k: int = 8
    source_filter: Optional[List[str]] = None


class ChunkPreview(BaseModel):
    source: str
    page: int
    chunk_type: str
    score: float
    text_preview: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    chunks: List[ChunkPreview]


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    global pipeline
    if pipeline is None:
        try:
            pipeline = RAGPipeline()
        except Exception as e:
            log.error(f"Pipeline init failed: {e}")
            raise HTTPException(status_code=503, detail="Collection not initialized.")
    log.info(f"Query: \"{request.question}\" | top_k={request.top_k} | filter={request.source_filter}")
    result = pipeline.query(
        question=request.question,
        top_k=request.top_k,
        source_filter=request.source_filter,
    )
    log.info(f"Answer generated | sources={result['sources']}")
    return result


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/sources")
def list_sources():
    from src.ingestion.vector_store import get_collection
    try:
        col = get_collection()
        results = col.get(include=["metadatas"])
        sources = sorted({m["source"] for m in results["metadatas"]})
        log.info(f"Listed {len(sources)} indexed sources")
        return {"sources": sources, "count": len(sources)}
    except Exception:
        return {"status": [], "count": 0, "detail": "No documents found."}
