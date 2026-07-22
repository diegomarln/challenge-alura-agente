"""Configuración y extracción segura de respuestas de Gemini."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI


class RagError(Exception):
    """Error base de los componentes de generación RAG."""


class RagConfigurationError(RagError):
    """La configuración requerida para el modelo no es válida."""


class RagGenerationError(RagError):
    """La respuesta del modelo no contiene texto utilizable."""


@dataclass(frozen=True)
class _AttributeFailure:
    """Representa un atributo que no se pudo leer sin exponer su detalle."""

    cause: Exception


_MISSING = object()


def create_chat_model(model_name: str | None = None) -> ChatGoogleGenerativeAI:
    """Construye el cliente de chat configurado, sin invocarlo."""
    resolved_model = model_name if model_name is not None else os.getenv("GEMINI_CHAT_MODEL")
    if not isinstance(resolved_model, str) or not resolved_model.strip():
        raise RagConfigurationError("Se requiere GEMINI_CHAT_MODEL o un model_name no vacío")
    try:
        return ChatGoogleGenerativeAI(model=resolved_model.strip())
    except Exception as error:
        raise RagConfigurationError("No se pudo crear el modelo de chat configurado") from error


def _extract_response_text(response: Any) -> str:
    """Extrae exclusivamente bloques textuales de una respuesta de modelo."""
    failures: list[Exception] = []
    response_text = _safe_getattr(response, "text")
    text = _text_or_record_failure(response_text, failures)
    if text is not None:
        return text

    content = _safe_getattr(response, "content")
    text = _text_or_record_failure(content, failures)
    if text is not None:
        return text
    if not isinstance(content, (list, tuple)):
        _raise_generation_error(failures)

    text_blocks: list[str] = []
    for block in content:
        if isinstance(block, dict):
            block_type = block.get("type")
            block_text = block.get("text")
        else:
            block_type = _safe_getattr(block, "type")
            block_text = _safe_getattr(block, "text")
        if isinstance(block_type, _AttributeFailure):
            failures.append(block_type.cause)
            continue
        if isinstance(block_text, _AttributeFailure):
            failures.append(block_text.cause)
            continue
        text = _useful_text(block_text) if block_type == "text" else None
        if text is not None:
            text_blocks.append(text)
    if text_blocks:
        return "".join(text_blocks)
    _raise_generation_error(failures)


def _useful_text(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _safe_getattr(obj: object, attribute: str) -> object | _AttributeFailure:
    """Lee un atributo sin permitir que errores de propiedades escapen."""
    try:
        return getattr(obj, attribute)
    except AttributeError:
        return _MISSING
    except Exception as error:
        return _AttributeFailure(error)


def _text_or_record_failure(value: object, failures: list[Exception]) -> str | None:
    if isinstance(value, _AttributeFailure):
        failures.append(value.cause)
        return None
    return _useful_text(value)


def _raise_generation_error(failures: list[Exception]) -> None:
    error = RagGenerationError("La respuesta del modelo no contiene texto utilizable")
    if failures:
        raise error from failures[0]
    raise error
