"""Pruebas de FAISS local sin acceso a Gemini."""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.rag import (
    EmptyCorpusError,
    FaissVectorStore,
    IndexingResult,
    InvalidVectorError,
    VectorStorePersistenceError,
    index_directory,
)
from src.rag import vector_store as vector_store_module


class FakeEmbeddings(Embeddings):
    """Embeddings deterministas basados en palabras conocidas."""

    model = "fake-embedding-v1"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    @staticmethod
    def _vector(text: str) -> list[float]:
        words = text.lower().split()
        return [float(words.count("gato")), float(words.count("perro")), float(words.count("cafe"))]


class InconsistentEmbeddings(FakeEmbeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0], [1.0, 0.0, 0.0]][: len(texts)]


class BadQueryEmbeddings(FakeEmbeddings):
    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


class ZeroDocumentEmbeddings(FakeEmbeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.0, 0.0, 0.0] for _ in texts]


class ZeroQueryEmbeddings(FakeEmbeddings):
    def embed_query(self, text: str) -> list[float]:
        return [0.0, 0.0, 0.0]


class NonFiniteEmbeddings(FakeEmbeddings):
    def __init__(self, value: float) -> None:
        self.value = value

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[self.value, 1.0, 0.0] for _ in texts]


def _documents() -> list[Document]:
    return [
        Document(page_content="gato gato", metadata={"source": "gatos.md", "page": 1}),
        Document(page_content="perro", metadata={"source": "perros.md", "slide": 2}),
        Document(page_content="cafe cafe", metadata={"source": "cafe.md", "sheet": "Datos"}),
    ]


def test_builds_store_and_searches_in_similarity_order() -> None:
    store = FaissVectorStore.from_documents(_documents(), FakeEmbeddings())

    results = store.similarity_search_with_score("gato", k=2)

    assert store.index.ntotal == 3
    assert results[0][0].metadata["source"] == "gatos.md"
    assert results == sorted(results, key=lambda item: item[1], reverse=True)
    assert all(isinstance(score, float) for _, score in results)


def test_search_limits_k_and_filters_by_threshold() -> None:
    store = FaissVectorStore.from_documents(_documents(), FakeEmbeddings())

    assert len(store.similarity_search("perro", k=99)) == 3
    assert [item[0].metadata["source"] for item in store.similarity_search_with_score("gato", score_threshold=0.9)] == ["gatos.md"]


@pytest.mark.parametrize("query,k", [("", 1), ("   ", 1), ("gato", 0)])
def test_rejects_invalid_search_arguments(query: str, k: int) -> None:
    store = FaissVectorStore.from_documents(_documents(), FakeEmbeddings())

    with pytest.raises(ValueError):
        store.similarity_search_with_score(query, k=k)


def test_rejects_empty_corpus_inconsistent_vectors_and_bad_query_dimension() -> None:
    with pytest.raises(EmptyCorpusError):
        FaissVectorStore.from_documents([], FakeEmbeddings())
    with pytest.raises(InvalidVectorError):
        FaissVectorStore.from_documents(_documents()[:2], InconsistentEmbeddings())
    store = FaissVectorStore.from_documents(_documents(), BadQueryEmbeddings())
    with pytest.raises(InvalidVectorError, match="dimensión"):
        store.similarity_search_with_score("gato")


def test_rejects_empty_content_and_zero_norm_vectors() -> None:
    with pytest.raises(EmptyCorpusError):
        FaissVectorStore.from_documents([Document(page_content="  ")], FakeEmbeddings())
    with pytest.raises(InvalidVectorError, match="norma"):
        FaissVectorStore.from_documents(_documents(), ZeroDocumentEmbeddings())
    store = FaissVectorStore.from_documents(_documents(), ZeroQueryEmbeddings())
    with pytest.raises(InvalidVectorError, match="norma"):
        store.similarity_search_with_score("consulta")


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_rejects_non_finite_document_vectors(value: float) -> None:
    with pytest.raises(InvalidVectorError, match="no finitos"):
        FaissVectorStore.from_documents(_documents(), NonFiniteEmbeddings(value))


def test_persistence_loads_contents_and_metadata_without_pickle(tmp_path: Path) -> None:
    store = FaissVectorStore.from_documents(_documents(), FakeEmbeddings())
    output = tmp_path / "vector_store"

    store.save(output)
    loaded = FaissVectorStore.load(output, FakeEmbeddings())

    assert {path.name for path in output.iterdir()} == {"index.faiss", "documents.json", "manifest.json"}
    assert loaded.documents == _documents()
    assert loaded.embedding_model == "fake-embedding-v1"
    assert loaded.similarity_search("cafe")[0].metadata["source"] == "cafe.md"
    assert not list(output.glob("*.pkl"))
    assert not list(output.glob("*.pickle"))


def test_load_rejects_missing_and_corrupt_json_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "store"
    FaissVectorStore.from_documents(_documents(), FakeEmbeddings()).save(output)
    (output / "documents.json").unlink()
    with pytest.raises(VectorStorePersistenceError, match="Faltan"):
        FaissVectorStore.load(output, FakeEmbeddings())

    FaissVectorStore.from_documents(_documents(), FakeEmbeddings()).save(output)
    (output / "documents.json").write_text("{", encoding="utf-8")
    with pytest.raises(VectorStorePersistenceError) as error:
        FaissVectorStore.load(output, FakeEmbeddings())
    assert error.value.__cause__ is not None


@pytest.mark.parametrize(
    "documents_data",
    [
        {"not": "a list"},
        ["not an object"],
        [{"metadata": {}}],
        [{"page_content": 1, "metadata": {}}],
        [{"page_content": "   ", "metadata": {}}],
        [{"page_content": "texto"}],
        [{"page_content": "texto", "metadata": []}],
    ],
)
def test_load_rejects_structurally_invalid_documents_json(tmp_path: Path, documents_data: object) -> None:
    output = tmp_path / "store"
    FaissVectorStore.from_documents(_documents(), FakeEmbeddings()).save(output)
    (output / "documents.json").write_text(json.dumps(documents_data), encoding="utf-8")

    with pytest.raises(VectorStorePersistenceError) as error:
        FaissVectorStore.load(output, FakeEmbeddings())
    assert error.value.__cause__ is not None

    FaissVectorStore.from_documents(_documents(), FakeEmbeddings()).save(output)
    (output / "manifest.json").write_text("{", encoding="utf-8")
    with pytest.raises(VectorStorePersistenceError) as error:
        FaissVectorStore.load(output, FakeEmbeddings())
    assert error.value.__cause__ is not None


@pytest.mark.parametrize(
    "mutation",
    [
        lambda manifest: manifest.update(schema_version=999),
        lambda manifest: manifest.update(document_count=999),
        lambda manifest: manifest.update(vector_dimension=999),
    ],
)
def test_load_rejects_inconsistent_manifest(tmp_path: Path, mutation: object) -> None:
    output = tmp_path / "store"
    FaissVectorStore.from_documents(_documents(), FakeEmbeddings()).save(output)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutation(manifest)  # type: ignore[operator]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(VectorStorePersistenceError):
        FaissVectorStore.load(output, FakeEmbeddings())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", True),
        ("document_count", True),
        ("vector_dimension", True),
        ("index_type", 1),
        ("index_type", None),
        ("embedding_model", 1),
        ("embedding_model", ""),
        ("embedding_model", "   "),
    ],
)
def test_load_rejects_invalid_manifest_field_types(tmp_path: Path, field: str, value: object) -> None:
    output = tmp_path / "store"
    FaissVectorStore.from_documents(_documents(), FakeEmbeddings()).save(output)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(VectorStorePersistenceError):
        FaissVectorStore.load(output, FakeEmbeddings())


def test_load_rejects_non_ip_index_even_if_manifest_declares_it(tmp_path: Path) -> None:
    output = tmp_path / "store"
    FaissVectorStore.from_documents(_documents(), FakeEmbeddings()).save(output)
    index = faiss.IndexFlatL2(3)
    index.add(np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32))
    faiss.write_index(index, str(output / "index.faiss"))
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["index_type"] = "IndexFlatL2"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(VectorStorePersistenceError):
        FaissVectorStore.load(output, FakeEmbeddings())


def test_repeated_save_replaces_complete_store_and_keeps_it_searchable(tmp_path: Path) -> None:
    output = tmp_path / "store"
    FaissVectorStore.from_documents(_documents(), FakeEmbeddings()).save(output)
    replacement = [Document(page_content="perro perro", metadata={"source": "nuevo.md"})]

    FaissVectorStore.from_documents(replacement, FakeEmbeddings()).save(output)
    loaded = FaissVectorStore.load(output, FakeEmbeddings())

    assert loaded.documents == replacement
    assert loaded.similarity_search("perro")[0].metadata["source"] == "nuevo.md"
    assert not list(output.glob(".vector-store-staging-*"))
    assert not list(output.glob(".vector-store-backups-*"))


def test_failed_publish_restores_previous_complete_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "store"
    original = FaissVectorStore.from_documents(_documents(), FakeEmbeddings())
    original.save(output)
    replacement = FaissVectorStore.from_documents([Document(page_content="perro perro", metadata={"source": "nuevo.md"})], FakeEmbeddings())
    real_replace = vector_store_module.os.replace
    failed = False

    def fail_during_documents_publish(source: str | Path, destination: str | Path) -> None:
        nonlocal failed
        source_path, destination_path = Path(source), Path(destination)
        if (
            not failed
            and source_path.parent.name.startswith(".vector-store-staging-")
            and destination_path == output / "documents.json"
        ):
            failed = True
            raise OSError("fallo de publicación simulado")
        real_replace(source, destination)

    monkeypatch.setattr(vector_store_module.os, "replace", fail_during_documents_publish)

    with pytest.raises(VectorStorePersistenceError) as error:
        replacement.save(output)

    assert error.value.__cause__ is not None
    loaded = FaissVectorStore.load(output, FakeEmbeddings())
    assert loaded.documents == _documents()
    assert loaded.similarity_search("gato")[0].metadata["source"] == "gatos.md"
    assert not list(output.glob(".vector-store-staging-*"))
    assert not list(output.glob(".vector-store-backups-*"))


def test_index_directory_uses_ingestion_and_writes_all_artifacts(tmp_path: Path) -> None:
    documents_directory = tmp_path / "documents"
    documents_directory.mkdir()
    (documents_directory / "uno.md").write_text("gato gato perro", encoding="utf-8")
    nested = documents_directory / "sub"
    nested.mkdir()
    (nested / "dos.md").write_text("cafe cafe", encoding="utf-8")
    output = tmp_path / "output"

    result = index_directory(documents_directory, output, FakeEmbeddings(), chunk_size=8, chunk_overlap=2)

    assert isinstance(result, IndexingResult)
    assert result.documents_loaded == 2
    assert result.chunks_created > result.documents_loaded
    assert result.index_size == result.chunks_created
    assert result.vector_dimension == 3
    assert {path.name for path in output.iterdir()} == {"index.faiss", "documents.json", "manifest.json"}


def test_index_directory_rejects_directory_without_useful_content(tmp_path: Path) -> None:
    documents_directory = tmp_path / "documents"
    documents_directory.mkdir()
    (documents_directory / "vacio.md").write_text("  \n", encoding="utf-8")

    with pytest.raises(EmptyCorpusError):
        index_directory(documents_directory, tmp_path / "output", FakeEmbeddings())
