"""Pruebas de fragmentación de documentos."""

import pytest
from langchain_core.documents import Document

from src.rag import split_documents


def test_splits_content_preserves_metadata_and_start_index() -> None:
    document = Document(
        page_content="uno dos tres cuatro cinco seis siete ocho",
        metadata={"page": 2, "slide": 1, "sheet": "Resumen", "source": "informe.md", "file_name": "informe.md", "file_type": "md"},
    )

    chunks = split_documents([document], chunk_size=14, chunk_overlap=3)

    assert len(chunks) > 1
    assert [chunk.metadata["chunk_index"] for chunk in chunks] == list(range(1, len(chunks) + 1))
    assert all(chunk.metadata["start_index"] >= 0 for chunk in chunks)
    assert all(chunk.metadata["page"] == 2 for chunk in chunks)
    assert all(chunk.metadata["source"] == "informe.md" for chunk in chunks)


def test_chunk_index_restarts_for_each_source_document_and_excludes_empty_chunks() -> None:
    documents = [
        Document(page_content="a b c d e f", metadata={"source": "uno.md"}),
        Document(page_content="   ", metadata={"source": "vacio.md"}),
        Document(page_content="g h i j k l", metadata={"source": "dos.md"}),
    ]

    chunks = split_documents(documents, chunk_size=5, chunk_overlap=1)

    assert all(chunk.page_content.strip() for chunk in chunks)
    assert [chunk.metadata["chunk_index"] for chunk in chunks if chunk.metadata["source"] == "uno.md"][0] == 1
    assert [chunk.metadata["chunk_index"] for chunk in chunks if chunk.metadata["source"] == "dos.md"][0] == 1


@pytest.mark.parametrize("chunk_size,chunk_overlap", [(0, 0), (10, -1), (10, 10), (10, 11)])
def test_rejects_invalid_chunk_parameters(chunk_size: int, chunk_overlap: int) -> None:
    with pytest.raises(ValueError):
        split_documents([], chunk_size, chunk_overlap)


def test_empty_input_and_output_are_deterministic() -> None:
    document = Document(page_content="uno dos tres cuatro cinco", metadata={"source": "a.md"})

    assert split_documents([]) == []
    assert split_documents([document], 8, 2) == split_documents([document], 8, 2)


def test_start_index_is_exact_and_original_document_is_not_mutated() -> None:
    document = Document(page_content="alpha beta gamma delta", metadata={"source": "original.md", "page": 1})
    original_content = document.page_content
    original_metadata = dict(document.metadata)

    chunks = split_documents([document], chunk_size=10, chunk_overlap=0)

    assert [(chunk.page_content, chunk.metadata["start_index"]) for chunk in chunks] == [
        ("alpha beta", 0),
        ("gamma", 11),
        ("delta", 17),
    ]
    assert document.page_content == original_content
    assert document.metadata == original_metadata
