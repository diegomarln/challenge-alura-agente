"""Coordinación básica de recuperación, contexto, prompt y generación RAG."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .context import SourceReference, build_context
from .generation import RagConfigurationError, RagError, RagGenerationError, _extract_response_text
from .prompts import build_rag_messages
from .vector_store import VectorStoreError

FALLBACK_MESSAGE = "No encontré información suficiente en los documentos internos para responder esta consulta."


class RagRetrievalError(RagError):
    """La recuperación no produjo resultados utilizables para el servicio RAG."""


@dataclass(frozen=True)
class RagResponse:
    """Resultado final de una consulta RAG sin conservar documentos ni contexto."""

    question: str
    answer: str
    sources: tuple[SourceReference, ...]
    documents_found: int
    used_fallback: bool


class RagService:
    """Coordina las piezas RAG existentes para responder una consulta individual.

    El ``fallback_message`` se conserva exactamente como se recibió, incluidos
    sus espacios exteriores, después de comprobar que contenga texto útil.
    """

    def __init__(
        self,
        vector_store: Any,
        chat_model: Any,
        top_k: int = 4,
        score_threshold: float | None = 0.3,
        fallback_message: str = FALLBACK_MESSAGE,
    ) -> None:
        if vector_store is None:
            raise RagConfigurationError("vector_store no puede ser None")
        if chat_model is None:
            raise RagConfigurationError("chat_model no puede ser None")
        if type(top_k) is not int or top_k <= 0:
            raise RagConfigurationError("top_k debe ser un entero mayor que cero")
        if score_threshold is not None:
            if isinstance(score_threshold, bool) or not isinstance(score_threshold, (int, float)):
                raise RagConfigurationError("score_threshold debe ser un número finito o None")
            if not math.isfinite(score_threshold) or not -1 <= score_threshold <= 1:
                raise RagConfigurationError("score_threshold debe estar entre -1 y 1")
        if not isinstance(fallback_message, str) or not fallback_message.strip():
            raise RagConfigurationError("fallback_message debe ser un texto no vacío")
        self.vector_store = vector_store
        self.chat_model = chat_model
        self.top_k = top_k
        self.score_threshold = None if score_threshold is None else float(score_threshold)
        self.fallback_message = fallback_message

    def answer(self, question: str) -> RagResponse:
        """Busca evidencia, genera una respuesta fundamentada o devuelve fallback."""
        if not isinstance(question, str):
            raise RagConfigurationError("La pregunta debe ser un texto no vacío")
        normalized_question = question.strip()
        if not normalized_question:
            raise RagConfigurationError("La pregunta debe ser un texto no vacío")
        try:
            results = self.vector_store.similarity_search_with_score(
                normalized_question,
                k=self.top_k,
                score_threshold=self.score_threshold,
            )
        except VectorStoreError:
            raise
        except Exception as error:
            raise RagRetrievalError("No fue posible recuperar documentos para la consulta") from error
        if not isinstance(results, list):
            raise RagRetrievalError("El resultado de recuperación no tiene un formato válido")
        if not results:
            return RagResponse(
                question=normalized_question,
                answer=self.fallback_message,
                sources=(),
                documents_found=0,
                used_fallback=True,
            )
        context_bundle = build_context(results)
        messages = build_rag_messages(context=context_bundle.context, question=normalized_question)
        try:
            model_response = self.chat_model.invoke(messages)
        except RagGenerationError:
            raise
        except Exception as error:
            raise RagGenerationError("No fue posible generar una respuesta para la consulta") from error
        answer = _extract_response_text(model_response)
        return RagResponse(
            question=normalized_question,
            answer=answer,
            sources=context_bundle.sources,
            documents_found=context_bundle.documents_found,
            used_fallback=False,
        )
