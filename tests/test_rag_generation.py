"""Pruebas de configuración y extracción de respuestas de Gemini."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import src.rag.generation as generation
from src.rag import (
    FaissVectorStore,
    RagConfigurationError,
    RagError,
    RagGenerationError,
    create_chat_model,
    split_documents,
)


def test_create_chat_model_uses_trimmed_explicit_name(monkeypatch: pytest.MonkeyPatch) -> None:
    received: dict[str, object] = {}

    class FakeChatModel:
        def __init__(self, **kwargs: object) -> None:
            received.update(kwargs)

    monkeypatch.setattr(generation, "ChatGoogleGenerativeAI", FakeChatModel)

    model = create_chat_model("  gemini-test  ")

    assert isinstance(model, FakeChatModel)
    assert received == {"model": "gemini-test"}
    assert "temperature" not in received


def test_create_chat_model_uses_environment_when_name_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    received: dict[str, object] = {}
    monkeypatch.setenv("GEMINI_CHAT_MODEL", "modelo-configurado")
    monkeypatch.setattr(generation, "ChatGoogleGenerativeAI", lambda **kwargs: received.update(kwargs) or kwargs)

    assert create_chat_model() == {"model": "modelo-configurado"}


@pytest.mark.parametrize("configured_name", ["", "   "])
def test_create_chat_model_rejects_blank_environment_name(
    monkeypatch: pytest.MonkeyPatch, configured_name: str
) -> None:
    monkeypatch.setenv("GEMINI_CHAT_MODEL", configured_name)

    with pytest.raises(RagConfigurationError):
        create_chat_model()


@pytest.mark.parametrize("model_name", [None, "", "   "])
def test_create_chat_model_rejects_missing_or_blank_name(monkeypatch: pytest.MonkeyPatch, model_name: str | None) -> None:
    monkeypatch.delenv("GEMINI_CHAT_MODEL", raising=False)

    with pytest.raises(RagConfigurationError):
        create_chat_model(model_name)


def test_create_chat_model_wraps_constructor_error_without_leaking_details(monkeypatch: pytest.MonkeyPatch) -> None:
    def failing_constructor(**kwargs: object) -> object:
        raise RuntimeError("detalle interno del proveedor")

    monkeypatch.setattr(generation, "ChatGoogleGenerativeAI", failing_constructor)

    with pytest.raises(RagConfigurationError) as error:
        create_chat_model("modelo")

    assert isinstance(error.value.__cause__, RuntimeError)
    assert "detalle interno" not in str(error.value)


def test_extracts_preferred_response_text() -> None:
    response = SimpleNamespace(text="Respuesta directa", content=[{"type": "text", "text": "alternativa"}])

    assert generation._extract_response_text(response) == "Respuesta directa"


def test_empty_response_text_falls_back_to_string_content() -> None:
    response = SimpleNamespace(text="  ", content="Contenido útil")

    assert generation._extract_response_text(response) == "Contenido útil"


def test_extracts_text_blocks_in_order_and_ignores_non_textual_blocks() -> None:
    response = SimpleNamespace(
        content=[
            {"type": "text", "text": "Primero "},
            {"type": "signature", "signature": "ignorada"},
            {"type": "image_url", "image_url": "ignorada"},
            {"type": "text", "text": "segundo"},
        ]
    )

    assert generation._extract_response_text(response) == "Primero segundo"


def test_extracts_object_blocks_and_preserves_unicode_and_newlines() -> None:
    response = SimpleNamespace(
        content=(
            SimpleNamespace(type="text", text="¡Hola, José!\n"),
            SimpleNamespace(type="tool_use", text="ignorado"),
            SimpleNamespace(type="text", text="¿Cómo estás?"),
        )
    )

    assert generation._extract_response_text(response) == "¡Hola, José!\n¿Cómo estás?"


class TextFailureWithContent:
    @property
    def text(self) -> str:
        raise RuntimeError("texto interno")

    @property
    def content(self) -> str:
        return "Contenido alternativo"


class BothResponsePropertiesFail:
    @property
    def text(self) -> str:
        raise RuntimeError("texto interno")

    @property
    def content(self) -> str:
        raise RuntimeError("contenido interno")


class FailingTypeBlock:
    @property
    def type(self) -> str:
        raise RuntimeError("tipo interno")

    @property
    def text(self) -> str:
        return "ignorado"


class FailingTextBlock:
    type = "text"

    @property
    def text(self) -> str:
        raise RuntimeError("texto de bloque interno")


def test_text_property_failure_falls_back_to_content() -> None:
    assert generation._extract_response_text(TextFailureWithContent()) == "Contenido alternativo"


def test_failed_response_properties_raise_generic_generation_error() -> None:
    with pytest.raises(RagGenerationError) as error:
        generation._extract_response_text(BothResponsePropertiesFail())

    assert isinstance(error.value.__cause__, RuntimeError)
    assert "interno" not in str(error.value)


@pytest.mark.parametrize("failing_block", [FailingTypeBlock(), FailingTextBlock()])
def test_failing_block_does_not_prevent_valid_text_blocks(failing_block: object) -> None:
    response = SimpleNamespace(content=[failing_block, {"type": "text", "text": "texto válido"}])

    assert generation._extract_response_text(response) == "texto válido"


def test_invalid_empty_and_non_text_blocks_are_ignored() -> None:
    response = SimpleNamespace(
        content=(
            {"type": "text", "text": "   "},
            {"type": "text", "text": 7},
            {"type": 7, "text": "ignorado"},
            SimpleNamespace(),
            {"type": "image", "text": "ignorado"},
            {"type": "text", "text": "primero "},
            {"type": "text", "text": "segundo"},
        )
    )

    assert generation._extract_response_text(response) == "primero segundo"


def test_all_invalid_or_failing_blocks_raise_generation_error() -> None:
    response = SimpleNamespace(content=[FailingTypeBlock(), FailingTextBlock(), {"type": "text", "text": " "}, SimpleNamespace()])

    with pytest.raises(RagGenerationError) as error:
        generation._extract_response_text(response)

    assert isinstance(error.value.__cause__, RuntimeError)
    assert "interno" not in str(error.value)


class ArbitraryStructure:
    def __str__(self) -> str:
        raise AssertionError("No se debe convertir la estructura completa a texto")


@pytest.mark.parametrize(
    "response",
    [
        SimpleNamespace(),
        SimpleNamespace(text="", content=[]),
        SimpleNamespace(content=[{"type": "signature", "text": "firma"}]),
        SimpleNamespace(content=ArbitraryStructure()),
    ],
)
def test_rejects_responses_without_text(response: object) -> None:
    with pytest.raises(RagGenerationError, match="no contiene texto"):
        generation._extract_response_text(response)


def test_existing_public_exports_remain_available() -> None:
    assert issubclass(RagConfigurationError, RagError)
    assert issubclass(RagGenerationError, RagError)
    assert callable(split_documents)
    assert hasattr(FaissVectorStore, "from_documents")
