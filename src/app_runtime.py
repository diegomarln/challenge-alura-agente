"""Inicialización reutilizable y segura de los recursos del núcleo RAG."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping
from typing import Any

from dotenv import load_dotenv

from src.rag import FaissVectorStore, RagService, create_chat_model, create_embeddings, index_directory

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_INDEX_ARTIFACTS = ("index.faiss", "documents.json", "manifest.json")


class AppRuntimeError(Exception):
    """Error base de la inicialización del runtime de la aplicación."""


class AppConfigurationError(AppRuntimeError):
    """La configuración de ejecución no es utilizable."""


class AppInitializationError(AppRuntimeError):
    """Los recursos RAG no se pudieron inicializar."""


@dataclass(frozen=True)
class AppConfig:
    """Configuración no secreta y validada para el runtime de la aplicación."""

    documents_dir: Path
    vector_store_dir: Path
    chat_model: str
    embedding_model: str
    top_k: int
    score_threshold: float
    chunk_size: int
    chunk_overlap: int


@dataclass(frozen=True)
class AppResources:
    """Recursos RAG listos para responder.

    ``documents_loaded`` cuenta unidades antes del chunking; ``chunks_created``
    cuenta los fragmentos almacenados en FAISS.
    """

    rag_service: RagService
    documents_loaded: int
    chunks_created: int
    vector_dimension: int
    index_rebuilt: bool


def load_app_config(environ: Mapping[str, str] | None = None) -> AppConfig:
    """Carga dotenv sin sobrescribir el entorno y valida solo datos no secretos."""
    load_dotenv(override=False)
    environment = os.environ if environ is None else environ
    _require_text(environment, "GOOGLE_API_KEY")
    chat_model = _require_text(environment, "GEMINI_CHAT_MODEL")
    embedding_model = _require_text(environment, "GEMINI_EMBEDDING_MODEL")
    documents_dir = _resolve_path(environment.get("DOCUMENTS_DIR", "documents"))
    vector_store_dir = _resolve_path(environment.get("VECTOR_STORE_DIR", "data/vector_store"))
    if not documents_dir.is_dir():
        raise AppConfigurationError("El directorio de documentos configurado no es válido")
    if vector_store_dir.exists() and not vector_store_dir.is_dir():
        raise AppConfigurationError("El directorio del índice configurado no es válido")
    if _paths_overlap(documents_dir, vector_store_dir):
        raise AppConfigurationError("El corpus y el directorio del índice no pueden superponerse")
    top_k = _positive_int(environment.get("RETRIEVER_TOP_K", "8"), "RETRIEVER_TOP_K")
    score_threshold = _score_threshold(environment.get("RETRIEVER_SCORE_THRESHOLD", "0.55"))
    chunk_size = _positive_int(environment.get("CHUNK_SIZE", "300"), "CHUNK_SIZE")
    chunk_overlap = _non_negative_int(environment.get("CHUNK_OVERLAP", "30"), "CHUNK_OVERLAP")
    if chunk_overlap >= chunk_size:
        raise AppConfigurationError("La configuración de fragmentación no es válida")
    return AppConfig(
        documents_dir=documents_dir,
        vector_store_dir=vector_store_dir,
        chat_model=chat_model,
        embedding_model=embedding_model,
        top_k=top_k,
        score_threshold=score_threshold,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def initialize_app_resources(config: AppConfig, force_reindex: bool = False) -> AppResources:
    """Construye o carga recursos RAG.

    Detecta cambios de modelo de embeddings, pero el formato actual no permite
    detectar cambios en documentos, ``chunk_size`` o ``chunk_overlap``. La
    mitigación manual es ``force_reindex=True``; la futura interfaz debe
    ofrecer una acción visible para reconstruir el índice.
    """
    if not isinstance(config, AppConfig):
        raise AppConfigurationError("La configuración de aplicación no es válida")
    if type(force_reindex) is not bool:
        raise AppConfigurationError("force_reindex debe ser booleano")

    index_is_complete = _inspect_index_artifacts(config.vector_store_dir)
    try:
        embeddings = create_embeddings(config.embedding_model)
        chat_model = create_chat_model(config.chat_model)
    except (KeyboardInterrupt, SystemExit, MemoryError):
        raise
    except Exception as error:
        raise AppInitializationError("No fue posible inicializar los proveedores RAG") from error

    index_rebuilt = force_reindex or not index_is_complete
    try:
        if index_rebuilt:
            store, documents_loaded, chunks_created, vector_dimension = _rebuild_and_load(config, embeddings)
        else:
            store = FaissVectorStore.load(config.vector_store_dir, embeddings)
            if store.embedding_model != config.embedding_model:
                store, documents_loaded, chunks_created, vector_dimension = _rebuild_and_load(config, embeddings)
                index_rebuilt = True
            else:
                documents_loaded = _count_document_units(store.documents)
                chunks_created = len(store.documents)
                vector_dimension = store.vector_dimension
        rag_service = RagService(store, chat_model, top_k=config.top_k, score_threshold=config.score_threshold)
    except (KeyboardInterrupt, SystemExit, MemoryError):
        raise
    except Exception as error:
        raise AppInitializationError("No fue posible preparar los recursos RAG") from error
    return AppResources(rag_service, documents_loaded, chunks_created, vector_dimension, index_rebuilt)


def _rebuild_and_load(config: AppConfig, embeddings: Any) -> tuple[FaissVectorStore, int, int, int]:
    indexing_result = index_directory(
        config.documents_dir,
        config.vector_store_dir,
        embeddings,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    store = FaissVectorStore.load(config.vector_store_dir, embeddings)
    if store.embedding_model != config.embedding_model:
        raise AppInitializationError("El índice reconstruido no es compatible con la configuración actual")
    return store, indexing_result.documents_loaded, indexing_result.chunks_created, indexing_result.vector_dimension


def _inspect_index_artifacts(directory: Path) -> bool:
    """Devuelve si están los tres archivos regulares; rechaza entradas inseguras."""
    regular_count = 0
    for artifact in _INDEX_ARTIFACTS:
        path = directory / artifact
        if path.is_symlink():
            raise AppInitializationError("Los artefactos del índice no son seguros")
        if not path.exists():
            continue
        if not path.is_file():
            raise AppInitializationError("Los artefactos del índice no son válidos")
        regular_count += 1
    return regular_count == len(_INDEX_ARTIFACTS)


def _count_document_units(documents: list[Any]) -> int:
    """Cuenta documentos previos al chunking a partir de metadatos de chunks."""
    identities: set[tuple[object, ...]] = set()
    for document in documents:
        metadata = getattr(document, "metadata", None)
        if not isinstance(metadata, Mapping):
            raise AppInitializationError("No se pudieron calcular las métricas del índice")
        source = metadata.get("source")
        if not isinstance(source, str) or not source.strip():
            raise AppInitializationError("No se pudieron calcular las métricas del índice")
        identity: list[object] = [source.strip()]
        for field in ("page", "slide"):
            if field in metadata:
                value = metadata[field]
                if type(value) is not int or value <= 0:
                    raise AppInitializationError("No se pudieron calcular las métricas del índice")
                identity.extend((field, value))
        if "sheet" in metadata:
            sheet = metadata["sheet"]
            if not isinstance(sheet, str) or not sheet.strip():
                raise AppInitializationError("No se pudieron calcular las métricas del índice")
            identity.extend(("sheet", sheet.strip()))
        identities.add(tuple(identity))
    return len(identities)


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first.is_relative_to(second) or second.is_relative_to(first)


def _resolve_path(value: object) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise AppConfigurationError("La configuración de rutas no es válida")
    path = Path(value.strip())
    return path.resolve() if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


def _require_text(environment: Mapping[str, str], name: str) -> str:
    value = environment.get(name)
    if not isinstance(value, str) or not value.strip():
        raise AppConfigurationError(f"Falta la variable requerida: {name}")
    return value.strip()


def _positive_int(value: object, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise AppConfigurationError(f"La configuración de {name} no es válida") from error
    if isinstance(value, bool) or parsed <= 0:
        raise AppConfigurationError(f"La configuración de {name} no es válida")
    return parsed


def _non_negative_int(value: object, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise AppConfigurationError(f"La configuración de {name} no es válida") from error
    if isinstance(value, bool) or parsed < 0:
        raise AppConfigurationError(f"La configuración de {name} no es válida")
    return parsed


def _score_threshold(value: object) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise AppConfigurationError("La configuración de RETRIEVER_SCORE_THRESHOLD no es válida") from error
    if isinstance(value, bool) or not math.isfinite(parsed) or not -1 <= parsed <= 1:
        raise AppConfigurationError("La configuración de RETRIEVER_SCORE_THRESHOLD no es válida")
    return parsed
