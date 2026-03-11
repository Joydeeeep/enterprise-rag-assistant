"""
Document loader: load and preprocess PDF, Markdown, and TXT from a folder.
"""

from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader

from utils.logger import get_logger


_SUPPORTED_EXTS: set[str] = {".pdf", ".md", ".markdown", ".txt"}


def _iter_files(root: Path, recursive: bool) -> Iterable[Path]:
    pattern = "**/*" if recursive else "*"
    for p in root.glob(pattern):
        if p.is_file():
            yield p


def _clean_text(text: str) -> str:
    # Keep this intentionally lightweight: normalize whitespace and strip null bytes.
    cleaned = text.replace("\x00", " ").strip()
    cleaned = " ".join(cleaned.split())
    return cleaned


def _dataset_metadata(root: Path, path: Path) -> dict[str, str]:
    """
    Compute stable, enterprise-style metadata for a file under the docs root.

    If files are organized as:
        <root>/<dataset_name>/<relative_path>
    then metadata is:
        {"source": "<dataset_name>", "file": "<relative_path>"}

    This matches the requested shape, e.g.:
        {"source": "docker_docs", "file": "engine/install.md"}
    """
    try:
        rel = path.relative_to(root)
        parts = rel.parts
        if len(parts) >= 2:
            dataset = parts[0]
            rel_file = "/".join(parts[1:])  # normalize to POSIX-like path for portability
            return {"source": dataset, "file": rel_file}
    except ValueError:
        # path isn't under root; fall back below
        pass
    return {"source": root.name, "file": path.name}


def load_documents(
    docs_dir: Path | str,
    *,
    recursive: bool = True,
    allowed_exts: Optional[Sequence[str]] = None,
) -> List[Document]:
    """
    Load PDF, Markdown, and TXT files from a directory into LangChain Documents.

    Args:
        docs_dir: Directory containing raw documents.
        recursive: If True, search subfolders.
        allowed_exts: Optional override list of allowed extensions (e.g. [".pdf", ".txt"]).

    Returns:
        List of LangChain `Document`s with cleaned text and metadata.
    """
    logger = get_logger()
    root = Path(docs_dir)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Docs directory not found: {root}")

    allowed = {e.lower() for e in (allowed_exts or _SUPPORTED_EXTS)}
    files = [p for p in _iter_files(root, recursive=recursive) if p.suffix.lower() in allowed]
    logger.info(f"Found {len(files)} documents under {root}")

    documents: List[Document] = []
    for path in files:
        ext = path.suffix.lower()
        try:
            if ext == ".pdf":
                loader = PyPDFLoader(str(path))
                loaded = loader.load()
            elif ext in {".md", ".markdown"}:
                # Markdown is plain text; keep parsing lightweight and stable.
                loader = TextLoader(str(path), autodetect_encoding=True)
                loaded = loader.load()
            elif ext == ".txt":
                loader = TextLoader(str(path), autodetect_encoding=True)
                loaded = loader.load()
            else:
                # Should be filtered, but keep this defensive.
                continue

            for d in loaded:
                content = _clean_text(d.page_content or "")
                if not content:
                    continue
                meta = dict(d.metadata or {})
                meta.update(_dataset_metadata(root, path))
                meta.update({"file_name": path.name, "file_ext": ext})
                documents.append(Document(page_content=content, metadata=meta))
        except Exception as e:  # noqa: BLE001
            logger.exception(f"Failed loading {path}: {e}")

    logger.info(f"Loaded {len(documents)} document pages/sections")
    return documents
