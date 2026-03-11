"""
FastAPI backend: POST /query for RAG questions.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.rag_pipeline import get_pipeline
from utils.logger import get_logger, setup_logger


setup_logger()
logger = get_logger()

app = FastAPI(
    title="Enterprise RAG Assistant API",
    version="0.1.0",
    description="FastAPI backend for a Retrieval-Augmented Generation assistant.",
)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    top_k: Optional[int] = Field(default=None, ge=1, le=20)


class SourceItem(BaseModel):
    rank: int
    score: float
    metadata: Dict[str, Any]
    text: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceItem]


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.on_event("startup")
def _startup() -> None:
    # Lazy: we only construct the pipeline (loads FAISS + embedder). LLM may load on first query,
    # depending on config; this keeps startup fast in dev and avoids accidental large downloads.
    try:
        _ = get_pipeline()
        logger.info("RAG pipeline initialized")
    except Exception as e:  # noqa: BLE001
        # Don't crash the server on startup; surface on query.
        logger.exception(f"Pipeline init failed: {e}")


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    try:
        pipeline = get_pipeline()
        answer, sources = pipeline.answer(req.question, top_k=req.top_k)
        return QueryResponse(answer=answer, sources=[SourceItem(**s) for s in sources])
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        logger.exception(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
