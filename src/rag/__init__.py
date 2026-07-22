"""Componentes locales para indexación y recuperación semántica."""

from .chunking import split_documents
from .context import ContextBundle, RagContextError, SourceReference, build_context
from .embeddings import create_embeddings
from .generation import RagConfigurationError, RagError, RagGenerationError, create_chat_model
from .prompts import RagPromptError, build_rag_messages
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
    "ContextBundle",
    "FaissVectorStore",
    "IndexingResult",
    "InvalidVectorError",
    "RagConfigurationError",
    "RagContextError",
    "RagError",
    "RagGenerationError",
    "RagPromptError",
    "VectorStoreError",
    "VectorStorePersistenceError",
    "SourceReference",
    "build_context",
    "build_rag_messages",
    "create_embeddings",
    "create_chat_model",
    "index_directory",
    "split_documents",
]
