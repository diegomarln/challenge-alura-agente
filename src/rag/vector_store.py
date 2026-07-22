"""Índice vectorial local FAISS con persistencia JSON segura."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import faiss
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.ingestion import load_documents

from .chunking import split_documents

SCHEMA_VERSION = 1
INDEX_FILENAME = "index.faiss"
DOCUMENTS_FILENAME = "documents.json"
MANIFEST_FILENAME = "manifest.json"


class VectorStoreError(Exception):
    """Error base para operaciones del almacén vectorial."""


class EmptyCorpusError(VectorStoreError):
    """El corpus no contiene documentos indexables."""


class InvalidVectorError(VectorStoreError):
    """Los vectores recibidos no son compatibles con el índice."""


class VectorStorePersistenceError(VectorStoreError):
    """La persistencia del índice no es válida o no se puede procesar."""


@dataclass(frozen=True)
class IndexingResult:
    documents_loaded: int
    chunks_created: int
    index_size: int
    vector_dimension: int
    output_directory: Path


class FaissVectorStore:
    """Índice de producto interno para similitud coseno sobre vectores normalizados."""

    def __init__(
        self,
        index: faiss.Index,
        documents: list[Document],
        embeddings: Embeddings,
        embedding_model: str | None = None,
    ) -> None:
        self.index = index
        self.documents = documents
        self.embeddings = embeddings
        self.embedding_model = embedding_model

    @classmethod
    def from_documents(
        cls,
        documents: list[Document],
        embeddings: Embeddings,
        embedding_model: str | None = None,
    ) -> "FaissVectorStore":
        if not documents:
            raise EmptyCorpusError("No hay documentos para indexar")
        contents = [document.page_content for document in documents]
        if any(not content or not content.strip() for content in contents):
            raise EmptyCorpusError("Los documentos deben contener texto no vacío")
        try:
            raw_vectors = embeddings.embed_documents(contents)
        except Exception as error:
            raise InvalidVectorError("No fue posible generar embeddings de documentos") from error
        vectors = _validate_vectors(raw_vectors, len(documents))
        index = faiss.IndexFlatIP(vectors.shape[1])
        faiss.normalize_L2(vectors)
        index.add(vectors)
        return cls(index, list(documents), embeddings, embedding_model or _embedding_model_name(embeddings))

    @property
    def vector_dimension(self) -> int:
        return int(self.index.d)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        score_threshold: float | None = None,
    ) -> list[tuple[Document, float]]:
        if not query or not query.strip():
            raise ValueError("La consulta no puede estar vacía")
        if k <= 0:
            raise ValueError("k debe ser mayor que cero")
        if score_threshold is not None and not isinstance(score_threshold, (int, float)):
            raise ValueError("score_threshold debe ser numérico o None")
        try:
            query_vector = np.asarray(self.embeddings.embed_query(query), dtype=np.float32)
        except Exception as error:
            raise InvalidVectorError("No fue posible generar el embedding de la consulta") from error
        if query_vector.ndim != 1 or query_vector.size != self.vector_dimension:
            raise InvalidVectorError("La dimensión del embedding de consulta no coincide con el índice")
        if not np.isfinite(query_vector).all():
            raise InvalidVectorError("El embedding de consulta contiene valores no finitos")
        query_norm = np.linalg.norm(query_vector)
        if not np.isfinite(query_norm) or query_norm == 0:
            raise InvalidVectorError("El embedding de consulta no puede tener norma L2 cero")
        query_vector = np.ascontiguousarray(query_vector.reshape(1, -1), dtype=np.float32)
        faiss.normalize_L2(query_vector)
        scores, indices = self.index.search(query_vector, min(k, len(self.documents)))
        results = [
            (self.documents[int(index)], float(score))
            for score, index in zip(scores[0], indices[0], strict=True)
            if 0 <= int(index) < len(self.documents)
            and (score_threshold is None or float(score) >= score_threshold)
        ]
        return sorted(results, key=lambda result: result[1], reverse=True)

    def similarity_search(self, query: str, k: int = 4, score_threshold: float | None = None) -> list[Document]:
        """Devuelve solo los documentos más similares."""
        return [document for document, _ in self.similarity_search_with_score(query, k, score_threshold)]

    def save(self, directory: str | Path) -> None:
        """Guarda los tres artefactos gestionados de forma recuperable."""
        output = Path(directory)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "document_count": len(self.documents),
            "vector_dimension": self.vector_dimension,
            "index_type": type(self.index).__name__,
            "embedding_model": self.embedding_model,
        }
        serialized_documents = [
            {"page_content": document.page_content, "metadata": document.metadata} for document in self.documents
        ]
        artifact_names = (INDEX_FILENAME, DOCUMENTS_FILENAME, MANIFEST_FILENAME)
        transaction_id = uuid4().hex
        staging = output / f".vector-store-staging-{transaction_id}"
        backups = output / f".vector-store-backups-{transaction_id}"
        published: set[str] = set()
        backed_up: set[str] = set()
        try:
            output.mkdir(parents=True, exist_ok=True)
            staging.mkdir()
            faiss.write_index(self.index, str(staging / INDEX_FILENAME))
            _write_json(staging / DOCUMENTS_FILENAME, serialized_documents)
            _write_json(staging / MANIFEST_FILENAME, manifest)

            backups.mkdir()
            for name in artifact_names:
                target = output / name
                if target.exists():
                    os.replace(target, backups / name)
                    backed_up.add(name)
            for name in artifact_names:
                os.replace(staging / name, output / name)
                published.add(name)
        except Exception as error:
            restoration_error: Exception | None = None
            try:
                for name in published - backed_up:
                    (output / name).unlink(missing_ok=True)
                for name in backed_up:
                    backup = backups / name
                    if backup.exists():
                        os.replace(backup, output / name)
            except Exception as restore_error:
                restoration_error = restore_error
            _cleanup_transaction_directory(staging, artifact_names)
            if restoration_error is None:
                _cleanup_transaction_directory(backups, artifact_names)
            else:
                error.add_note(f"También falló la restauración del estado anterior: {restoration_error}")
                raise VectorStorePersistenceError(
                    f"No se pudo guardar el índice en '{output}' y no se pudo restaurar el estado anterior"
                ) from error
            raise VectorStorePersistenceError(f"No se pudo guardar el índice en '{output}'") from error
        _cleanup_transaction_directory(staging, artifact_names)
        _cleanup_transaction_directory(backups, artifact_names)

    @classmethod
    def load(cls, directory: str | Path, embeddings: Embeddings) -> "FaissVectorStore":
        """Carga y valida un índice FAISS y sus metadatos JSON."""
        source = Path(directory)
        paths = {name: source / name for name in (INDEX_FILENAME, DOCUMENTS_FILENAME, MANIFEST_FILENAME)}
        missing = [name for name, path in paths.items() if not path.is_file()]
        if missing:
            raise VectorStorePersistenceError(f"Faltan artefactos requeridos: {', '.join(missing)}")
        try:
            documents_data = _read_json(paths[DOCUMENTS_FILENAME])
            manifest = _read_json(paths[MANIFEST_FILENAME])
            if not isinstance(documents_data, list) or not isinstance(manifest, dict):
                raise ValueError("La estructura JSON no es válida")
            documents = [_document_from_json(item) for item in documents_data]
            index = faiss.read_index(str(paths[INDEX_FILENAME]))
            _validate_loaded(index, documents, manifest)
        except VectorStorePersistenceError:
            raise
        except Exception as error:
            raise VectorStorePersistenceError(f"No se pudo cargar el índice desde '{source}'") from error
        return cls(index, documents, embeddings, manifest.get("embedding_model"))


def index_directory(
    documents_directory: str | Path,
    vector_store_directory: str | Path,
    embeddings: Embeddings,
    chunk_size: int = 300,
    chunk_overlap: int = 30,
) -> IndexingResult:
    """Carga, fragmenta, indexa y persiste los documentos de un directorio."""
    documents = load_documents(documents_directory)
    if not documents:
        raise EmptyCorpusError("No se encontraron documentos con contenido útil")
    chunks = split_documents(documents, chunk_size, chunk_overlap)
    if not chunks:
        raise EmptyCorpusError("No se generaron fragmentos útiles")
    store = FaissVectorStore.from_documents(chunks, embeddings)
    output = Path(vector_store_directory)
    store.save(output)
    return IndexingResult(len(documents), len(chunks), int(store.index.ntotal), store.vector_dimension, output)


def _validate_vectors(raw_vectors: Any, expected_count: int) -> np.ndarray:
    try:
        vectors = np.asarray(raw_vectors, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise InvalidVectorError("Los embeddings no se pueden convertir a float32") from error
    if vectors.ndim != 2 or vectors.shape[0] != expected_count:
        raise InvalidVectorError("La cantidad de embeddings no coincide con los documentos")
    if vectors.shape[1] <= 0:
        raise InvalidVectorError("Los embeddings deben tener una dimensión mayor que cero")
    if not np.isfinite(vectors).all():
        raise InvalidVectorError("Los embeddings contienen valores no finitos")
    norms = np.linalg.norm(vectors, axis=1)
    if not np.isfinite(norms).all() or np.any(norms == 0):
        raise InvalidVectorError("Los embeddings no pueden contener vectores de norma L2 cero")
    return np.ascontiguousarray(vectors, dtype=np.float32)


def _embedding_model_name(embeddings: Embeddings) -> str | None:
    value = getattr(embeddings, "model", None)
    return value if isinstance(value, str) and value else None


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(value, file, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _document_from_json(value: Any) -> Document:
    if not isinstance(value, dict) or set(value) != {"page_content", "metadata"}:
        raise ValueError("Entrada de documento inválida")
    content, metadata = value["page_content"], value["metadata"]
    if not isinstance(content, str) or not content.strip() or not isinstance(metadata, dict):
        raise ValueError("Contenido o metadatos de documento inválidos")
    return Document(page_content=content, metadata=metadata)


def _validate_loaded(index: faiss.Index, documents: list[Document], manifest: dict[str, Any]) -> None:
    schema_version = manifest.get("schema_version")
    document_count = manifest.get("document_count")
    vector_dimension = manifest.get("vector_dimension")
    index_type = manifest.get("index_type")
    embedding_model = manifest.get("embedding_model")
    if type(schema_version) is not int or schema_version != SCHEMA_VERSION:
        raise ValueError("Versión de esquema no compatible")
    if type(document_count) is not int or document_count <= 0:
        raise ValueError("document_count inválido")
    if type(vector_dimension) is not int or vector_dimension <= 0:
        raise ValueError("vector_dimension inválida")
    if not isinstance(index_type, str) or index_type != "IndexFlatIP":
        raise ValueError("El tipo de índice del manifiesto debe ser IndexFlatIP")
    if embedding_model is not None and (not isinstance(embedding_model, str) or not embedding_model.strip()):
        raise ValueError("embedding_model inválido")
    if not isinstance(index, faiss.IndexFlatIP):
        raise ValueError("El índice persistido debe ser IndexFlatIP")
    if document_count != len(documents) or document_count != index.ntotal:
        raise ValueError("La cantidad de documentos no coincide con el índice")
    if vector_dimension != index.d:
        raise ValueError("La dimensión del índice no coincide con el manifiesto")


def _cleanup_transaction_directory(directory: Path, artifact_names: tuple[str, str, str]) -> None:
    """Elimina solo artefactos temporales conocidos de una transacción propia."""
    for name in artifact_names:
        (directory / name).unlink(missing_ok=True)
    try:
        directory.rmdir()
    except OSError:
        pass
