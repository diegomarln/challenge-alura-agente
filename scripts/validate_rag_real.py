"""Valida manualmente el núcleo RAG con los documentos y modelos configurados localmente."""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from tempfile import TemporaryDirectory
from typing import Sequence

from dotenv import load_dotenv

from src.ingestion import DocumentLoadError
from src.rag import (
    FaissVectorStore,
    RagConfigurationError,
    RagGenerationError,
    RagResponse,
    RagRetrievalError,
    RagService,
    VectorStoreError,
    create_chat_model,
    create_embeddings,
    index_directory,
)

DEFAULT_QUESTION = "¿Cuál es el monto máximo mensual que se puede reembolsar por internet en casa?"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
UNKNOWN_SOURCE_NAME = "fuente no identificada"


class ValidationConfigurationError(Exception):
    """La configuración o los argumentos de la validación local no son válidos."""


class ProviderInitializationError(Exception):
    """Un proveedor configurado no pudo inicializarse sin exponer sus detalles."""


@dataclass(frozen=True)
class RuntimeConfig:
    """Valores no secretos necesarios para ejecutar una validación local."""

    question: str
    documents_directory: Path
    top_k: int
    score_threshold: float
    chunk_size: int
    chunk_overlap: int
    chat_model: str
    embedding_model: str


@dataclass(frozen=True)
class ValidationResult:
    """Resultado seguro que puede mostrarse por la interfaz de línea de comandos."""

    documents_loaded: int
    chunks_created: int
    vector_dimension: int
    response: RagResponse


def build_parser() -> argparse.ArgumentParser:
    """Construye el parser sin leer configuración ni inicializar proveedores."""
    parser = argparse.ArgumentParser(description="Valida localmente el flujo RAG con Gemini configurado.")
    parser.add_argument("--question", default=DEFAULT_QUESTION, help="Pregunta para el agente RAG.")
    parser.add_argument("--documents-dir", default=None, help="Directorio de documentos compatible con la ingesta.")
    parser.add_argument("--top-k", default=None, help="Cantidad de fragmentos recuperados.")
    parser.add_argument("--score-threshold", default=None, help="Umbral de similitud entre -1 y 1.")
    parser.add_argument("--chunk-size", default=None, help="Tamaño de cada fragmento.")
    parser.add_argument("--chunk-overlap", default=None, help="Superposición entre fragmentos.")
    return parser


def load_runtime_config(args: argparse.Namespace, environ: dict[str, str] | None = None) -> RuntimeConfig:
    """Carga dotenv al ejecutar y valida parámetros sin exponer valores secretos."""
    load_dotenv()
    environment = os.environ if environ is None else environ
    _require_text(environment, "GOOGLE_API_KEY")
    chat_model = _require_text(environment, "GEMINI_CHAT_MODEL")
    embedding_model = _require_text(environment, "GEMINI_EMBEDDING_MODEL")

    question = _require_argument_text(args.question, "question")
    documents_directory = _resolve_documents_directory(args.documents_dir or environment.get("DOCUMENTS_DIR", "documents"))
    if not documents_directory.is_dir():
        raise ValidationConfigurationError("documents-dir debe existir y ser un directorio")
    top_k = _positive_int(args.top_k if args.top_k is not None else environment.get("RETRIEVER_TOP_K", "4"), "top-k")
    score_threshold = _threshold(
        args.score_threshold if args.score_threshold is not None else environment.get("RETRIEVER_SCORE_THRESHOLD", "0.3")
    )
    chunk_size = _positive_int(args.chunk_size if args.chunk_size is not None else environment.get("CHUNK_SIZE", "300"), "chunk-size")
    chunk_overlap = _non_negative_int(
        args.chunk_overlap if args.chunk_overlap is not None else environment.get("CHUNK_OVERLAP", "30"), "chunk-overlap"
    )
    if chunk_overlap >= chunk_size:
        raise ValidationConfigurationError("chunk-overlap debe ser menor que chunk-size")
    return RuntimeConfig(
        question=question,
        documents_directory=documents_directory,
        top_k=top_k,
        score_threshold=score_threshold,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        chat_model=chat_model,
        embedding_model=embedding_model,
    )


def run_validation(config: RuntimeConfig) -> ValidationResult:
    """Ejecuta el flujo completo usando un índice temporal que se elimina al finalizar."""
    try:
        embeddings = create_embeddings(config.embedding_model)
    except Exception as error:
        raise ProviderInitializationError("No fue posible inicializar los embeddings.") from error
    try:
        chat_model = create_chat_model(config.chat_model)
    except Exception as error:
        raise ProviderInitializationError("No fue posible inicializar el modelo de chat.") from error
    with TemporaryDirectory(prefix="challenge-alura-rag-") as temporary_directory:
        indexing_result = index_directory(
            config.documents_directory,
            Path(temporary_directory),
            embeddings,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
        store = FaissVectorStore.load(Path(temporary_directory), embeddings)
        response = RagService(store, chat_model, top_k=config.top_k, score_threshold=config.score_threshold).answer(config.question)
        return ValidationResult(
            documents_loaded=indexing_result.documents_loaded,
            chunks_created=indexing_result.chunks_created,
            vector_dimension=indexing_result.vector_dimension,
            response=response,
        )


def format_source(source: object) -> str:
    """Formatea solo los campos seguros de una referencia pública."""
    file_name = getattr(source, "file_name", None)
    page = getattr(source, "page", None)
    chunk_index = getattr(source, "chunk_index", None)
    score = getattr(source, "score", None)
    parts = [f"archivo={_safe_source_name(file_name)}"]
    if type(page) is int:
        parts.append(f"página={page}")
    if type(chunk_index) is int:
        parts.append(f"fragmento={chunk_index}")
    if isinstance(score, (int, float)) and not isinstance(score, bool) and math.isfinite(score):
        parts.append(f"score={float(score):.3f}")
    return ", ".join(parts)


def format_summary(result: ValidationResult) -> str:
    """Crea una salida breve sin contexto, prompts, vectores ni rutas absolutas."""
    response = result.response
    lines = [
        f"Documentos cargados: {result.documents_loaded}",
        f"Fragmentos creados: {result.chunks_created}",
        f"Dimensión del índice: {result.vector_dimension}",
        f"Pregunta: {response.question}",
        f"Respuesta: {response.answer}",
        f"Fallback usado: {'sí' if response.used_fallback else 'no'}",
        f"Fuentes: {len(response.sources)}",
    ]
    lines.extend(f"- {format_source(source)}" for source in response.sources)
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Convierte errores de las capas en mensajes de CLI sin trazas ni secretos."""
    try:
        config = load_runtime_config(build_parser().parse_args(argv))
        print(format_summary(run_validation(config)))
        return 0
    except ValidationConfigurationError as error:
        print(f"Error de configuración: {error}", file=sys.stderr)
    except ProviderInitializationError as error:
        print(f"Error de proveedor: {error}", file=sys.stderr)
    except RagConfigurationError:
        print("Error de configuración del componente RAG.", file=sys.stderr)
    except DocumentLoadError:
        print("Error al cargar los documentos configurados.", file=sys.stderr)
    except VectorStoreError:
        print("Error al construir o consultar el índice vectorial.", file=sys.stderr)
    except RagRetrievalError:
        print("Error al recuperar documentos para la consulta.", file=sys.stderr)
    except RagGenerationError:
        print("Error al generar la respuesta del modelo.", file=sys.stderr)
    except Exception:
        print("La validación local no pudo completarse.", file=sys.stderr)
    return 1


def _require_text(environment: dict[str, str] | os._Environ[str], name: str) -> str:
    value = environment.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValidationConfigurationError(f"Falta la variable requerida: {name}")
    return value.strip()


def _require_argument_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationConfigurationError(f"{name} debe ser texto no vacío")
    return value.strip()


def _resolve_documents_directory(value: object) -> Path:
    """Ancla rutas relativas de CLI y entorno a la raíz del repositorio, no al CWD."""
    if not isinstance(value, str) or not value.strip():
        raise ValidationConfigurationError("documents-dir debe existir y ser un directorio")
    candidate = Path(value.strip())
    return candidate.resolve() if candidate.is_absolute() else (REPOSITORY_ROOT / candidate).resolve()


def _safe_source_name(value: object) -> str:
    """Devuelve solo un nombre de archivo seguro, sin rutas ni controles de salida."""
    if not isinstance(value, str):
        return UNKNOWN_SOURCE_NAME
    candidate = value.strip()
    if not candidate or any(ord(character) < 32 or ord(character) == 127 for character in candidate):
        return UNKNOWN_SOURCE_NAME
    posix_path = PurePosixPath(candidate)
    windows_path = PureWindowsPath(candidate)
    if posix_path.is_absolute() or windows_path.is_absolute() or windows_path.drive:
        return UNKNOWN_SOURCE_NAME
    name = candidate.replace("\\", "/").rsplit("/", maxsplit=1)[-1]
    return name if name and name not in {".", ".."} else UNKNOWN_SOURCE_NAME


def _positive_int(value: object, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ValidationConfigurationError(f"{name} debe ser un entero mayor que cero") from error
    if isinstance(value, bool) or parsed <= 0:
        raise ValidationConfigurationError(f"{name} debe ser un entero mayor que cero")
    return parsed


def _non_negative_int(value: object, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ValidationConfigurationError(f"{name} debe ser un entero no negativo") from error
    if isinstance(value, bool) or parsed < 0:
        raise ValidationConfigurationError(f"{name} debe ser un entero no negativo")
    return parsed


def _threshold(value: object) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValidationConfigurationError("score-threshold debe ser un número finito entre -1 y 1") from error
    if isinstance(value, bool) or not math.isfinite(parsed) or not -1 <= parsed <= 1:
        raise ValidationConfigurationError("score-threshold debe ser un número finito entre -1 y 1")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
