"""Referencias de fuentes y contexto documental delimitado para RAG."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath, PureWindowsPath

from langchain_core.documents import Document

from .generation import RagError


class RagContextError(RagError):
    """Los resultados recuperados no permiten construir contexto seguro."""


@dataclass(frozen=True)
class SourceReference:
    """Metadatos normalizados de una fuente recuperada."""

    reference: str
    source: str | None
    file_name: str | None
    file_type: str | None
    score: float
    page: int | None
    slide: int | None
    sheet: str | None
    chunk_index: int | None


@dataclass(frozen=True)
class ContextBundle:
    """Contexto delimitado y referencias asociadas a los resultados recuperados."""

    context: str
    sources: tuple[SourceReference, ...]
    documents_found: int


def build_context(results: Sequence[tuple[Document, float]]) -> ContextBundle:
    """Convierte resultados ordenados de búsqueda en contexto y referencias deterministas."""
    if not isinstance(results, Sequence):
        raise RagContextError("Los resultados recuperados deben ser una secuencia")
    if not results:
        return ContextBundle(context="", sources=(), documents_found=0)

    sources: list[SourceReference] = []
    sections: list[str] = []
    for position, item in enumerate(results, start=1):
        document, score = _validate_result(item)
        metadata = document.metadata if isinstance(document.metadata, Mapping) else {}
        reference = f"Fuente {position}"
        source_reference = SourceReference(
            reference=reference,
            source=_safe_identifier(metadata.get("source")),
            file_name=_safe_identifier(metadata.get("file_name")),
            file_type=_safe_text(metadata.get("file_type")),
            score=float(score),
            page=_positive_int(metadata.get("page")),
            slide=_positive_int(metadata.get("slide")),
            sheet=_safe_text(metadata.get("sheet")),
            chunk_index=_positive_int(metadata.get("chunk_index")),
        )
        sources.append(source_reference)
        sections.append(_format_section(document.page_content, source_reference))
    return ContextBundle(context="\n\n".join(sections), sources=tuple(sources), documents_found=len(sources))


def _validate_result(item: object) -> tuple[Document, float]:
    if not isinstance(item, tuple) or len(item) != 2:
        raise RagContextError("Cada resultado recuperado debe ser un par Document y score")
    document, score = item
    if not isinstance(document, Document):
        raise RagContextError("Cada resultado debe contener un Document")
    if not isinstance(document.page_content, str) or not document.page_content.strip():
        raise RagContextError("Los documentos recuperados deben contener texto no vacío")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
        raise RagContextError("El score recuperado debe ser un número finito")
    return document, float(score)


def _safe_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _safe_identifier(value: object) -> str | None:
    identifier = _safe_text(value)
    if identifier is None or _is_absolute_path(identifier):
        return None
    return identifier


def _is_absolute_path(value: str) -> bool:
    """Detecta rutas absolutas POSIX, Windows y UNC sin consultar el sistema."""
    return PureWindowsPath(value).is_absolute() or PurePosixPath(value).is_absolute()


def _positive_int(value: object) -> int | None:
    return value if type(value) is int and value > 0 else None


def _format_section(content: str, source: SourceReference) -> str:
    metadata_lines = ["Metadatos:"]
    if source.source is not None:
        metadata_lines.append(f"- Fuente: {source.source}")
    elif source.file_name is not None:
        metadata_lines.append(f"- Archivo: {source.file_name}")
    if source.file_type is not None:
        metadata_lines.append(f"- Tipo: {source.file_type}")
    if source.page is not None:
        metadata_lines.append(f"- Página: {source.page}")
    if source.slide is not None:
        metadata_lines.append(f"- Diapositiva: {source.slide}")
    if source.sheet is not None:
        metadata_lines.append(f"- Hoja: {source.sheet}")
    if source.chunk_index is not None:
        metadata_lines.append(f"- Fragmento: {source.chunk_index}")
    upper_reference = source.reference.upper()
    return "\n".join(
        [
            f"[INICIO {upper_reference}]",
            *metadata_lines,
            "",
            "Contenido:",
            "Contenido documental; cada línea está prefijada con | :",
            _quote_content(content),
            f"[FIN {upper_reference}]",
        ]
    )


def _quote_content(content: str) -> str:
    """Cita cada línea sin alterar sus terminadores ni espacios originales."""
    return "".join(f"| {line}" for line in content.splitlines(keepends=True))
