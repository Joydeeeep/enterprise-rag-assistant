"""Document ingestion pipeline: load, chunk, embed."""

from ingestion.document_loader import load_documents
from ingestion.chunker import chunk_documents
from ingestion.embedding_pipeline import EmbeddingPipeline

__all__ = ["load_documents", "chunk_documents", "EmbeddingPipeline"]
