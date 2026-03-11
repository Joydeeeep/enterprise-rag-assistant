"""
Chunking pipeline: split documents using RecursiveCharacterTextSplitter.
Chunk size ~500, overlap ~100.
"""

from typing import List, Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.logger import get_logger


def chunk_documents(
    documents: List[Document],
    *,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    separators: Sequence[str] | None = None,
) -> List[Document]:
    """
    Split documents into smaller chunks for embedding/retrieval.

    Notes:
        LangChain's RecursiveCharacterTextSplitter splits by characters, not tokens.
        With typical English text this approximates token counts, and can be swapped later
        for a token-aware splitter if desired.

    Args:
        documents: Input Documents.
        chunk_size: Target chunk size (characters).
        chunk_overlap: Overlap between chunks (characters).
        separators: Optional custom separators list.

    Returns:
        List of chunked Documents with inherited metadata.
    """
    logger = get_logger()
    if not documents:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=list(separators) if separators is not None else None,
        add_start_index=True,
    )

    chunks = splitter.split_documents(documents)
    logger.info(f"Chunked {len(documents)} docs into {len(chunks)} chunks")
    return chunks
