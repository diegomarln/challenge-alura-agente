"""Utilidades para cargar documentos corporativos."""

from .document_loader import (
    SUPPORTED_EXTENSIONS,
    DocumentLoadError,
    UnsupportedDocumentTypeError,
    load_document,
    load_documents,
)

__all__ = [
    "SUPPORTED_EXTENSIONS",
    "DocumentLoadError",
    "UnsupportedDocumentTypeError",
    "load_document",
    "load_documents",
]
