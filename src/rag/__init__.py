"""Componentes locales para indexación y recuperación semántica."""

from .chunking import split_documents
from .embeddings import create_embeddings
from .generation import RagConfigurationError, RagError, RagGenerationError, create_chat_model
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
    "RagConfigurationError",
    "RagError",
    "RagGenerationError",
    "VectorStoreError",
    "VectorStorePersistenceError",
    "create_embeddings",
    "create_chat_model",
    "index_directory",
    "split_documents",
]
