"""Fragmentación determinista de documentos para recuperación."""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(
    documents: Sequence[Document],
    chunk_size: int = 300,
    chunk_overlap: int = 30,
) -> list[Document]:
    """Divide documentos, conservando sus metadatos y posición de inicio."""
    if chunk_size <= 0:
        raise ValueError("chunk_size debe ser mayor que cero")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap no puede ser negativo")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap debe ser menor que chunk_size")
    if not documents:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    chunks: list[Document] = []
    for document in documents:
        chunk_index = 0
        for chunk in splitter.split_documents([document]):
            if not chunk.page_content.strip():
                continue
            chunk_index += 1
            metadata = dict(chunk.metadata)
            metadata["chunk_index"] = chunk_index
            chunks.append(Document(page_content=chunk.page_content, metadata=metadata))
    return chunks
