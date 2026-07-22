"""Fábrica de embeddings Gemini sin efectos de red durante la importación."""

from __future__ import annotations

import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings


def create_embeddings(model_name: str | None = None) -> GoogleGenerativeAIEmbeddings:
    """Construye embeddings Gemini usando el modelo indicado o la configuración de entorno."""
    resolved_model = model_name if model_name is not None else os.getenv("GEMINI_EMBEDDING_MODEL")
    if not resolved_model or not resolved_model.strip():
        raise ValueError("Se requiere GEMINI_EMBEDDING_MODEL o un model_name no vacío")
    return GoogleGenerativeAIEmbeddings(model=resolved_model.strip())
