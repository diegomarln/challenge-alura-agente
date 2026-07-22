"""Componentes locales para indexación y recuperación semántica."""

from .chunking import split_documents
from .embeddings import create_embeddings
from .vector_store import (
    EmptyCorpusError,
    FaissVectorStore,
    IndexingResult,
    InvalidVectorError,
    VectorStoreError,
    VectorStorePersistenceError,
    index_directory,
)

__all__ = [
    "EmptyCorpusError",
    "FaissVectorStore",
    "IndexingResult",
    "InvalidVectorError",
    "VectorStoreError",
    "VectorStorePersistenceError",
    "create_embeddings",
    "index_directory",
    "split_documents",
]
