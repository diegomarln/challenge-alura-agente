"""Pruebas focalizadas del servicio RAG básico mediante dobles deterministas."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from src.rag import (
    FALLBACK_MESSAGE,
    FaissVectorStore,
    RagConfigurationError,
    RagContextError,
    RagGenerationError,
    RagPromptError,
    RagResponse,
    RagRetrievalError,
    RagService,
    VectorStoreError,
    build_context,
    build_rag_messages,
    create_chat_model,
)


class FakeVectorStore:
    def __init__(self, results: object, error: Exception | None = None) -> None:
        self.results = results
        self.error = error
        self.calls: list[tuple[str, int, float | None]] = []

    def similarity_search_with_score(self, query: str, k: int, score_threshold: float | None) -> object:
        self.calls.append((query, k, score_threshold))
        if self.error is not None:
            raise self.error
        return self.results


class FakeChatModel:
    def __init__(self, response: object = SimpleNamespace(content="Respuesta generada"), error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[tuple[object, ...]] = []

    def invoke(self, messages: tuple[object, ...]) -> object:
        self.calls.append(messages)
        if self.error is not None:
            raise self.error
        return self.response


class StripCountingString(str):
    """Cadena que permite verificar que la normalización se ejecuta una vez."""

    def __new__(cls, value: str) -> StripCountingString:
        instance = super().__new__(cls, value)
        instance.strip_calls = 0
        return instance

    def strip(self, chars: str | None = None) -> str:
        self.strip_calls += 1
        return super().strip(chars)


def _results() -> list[tuple[Document, float]]:
    return [
        (Document(page_content="Política de café", metadata={"file_name": "politica.md", "page": 1}), 0.9),
        (Document(page_content="Segundo documento", metadata={"file_name": "dos.md", "page": 2}), 0.8),
    ]


def test_constructor_accepts_none_threshold_without_calling_dependencies() -> None:
    vector_store = FakeVectorStore([])
    model = FakeChatModel()

    service = RagService(vector_store, model, top_k=2, score_threshold=None)

    assert service.score_threshold is None
    assert vector_store.calls == []
    assert model.calls == []


@pytest.mark.parametrize(("value", "expected"), [(0, 0.0), (-1, -1.0), (1, 1.0), (0.25, 0.25)])
def test_threshold_is_normalized_to_float_and_forwarded(value: int | float, expected: float) -> None:
    vector_store = FakeVectorStore([])
    service = RagService(vector_store, FakeChatModel(), score_threshold=value)

    service.answer("consulta")

    assert service.score_threshold == expected
    assert isinstance(service.score_threshold, float)
    assert vector_store.calls == [("consulta", 4, expected)]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"vector_store": None},
        {"chat_model": None},
        {"top_k": 0},
        {"top_k": -1},
        {"top_k": True},
        {"score_threshold": True},
        {"score_threshold": float("nan")},
        {"score_threshold": float("inf")},
        {"score_threshold": float("-inf")},
        {"score_threshold": 1.1},
        {"score_threshold": -1.1},
        {"fallback_message": ""},
        {"fallback_message": "   "},
    ],
)
def test_constructor_rejects_invalid_configuration(kwargs: dict[str, object]) -> None:
    defaults: dict[str, object] = {"vector_store": FakeVectorStore([]), "chat_model": FakeChatModel()}
    defaults.update(kwargs)

    with pytest.raises(RagConfigurationError):
        RagService(**defaults)  # type: ignore[arg-type]


def test_fallback_message_preserves_outer_spaces_exactly() -> None:
    service = RagService(FakeVectorStore([]), FakeChatModel(), fallback_message="  Sin evidencia  ")

    response = service.answer("consulta")

    assert service.fallback_message == "  Sin evidencia  "
    assert response.answer == "  Sin evidencia  "


@pytest.mark.parametrize("question", [None, "", "   ", 7])
def test_answer_rejects_invalid_question(question: object) -> None:
    with pytest.raises(RagConfigurationError):
        RagService(FakeVectorStore([]), FakeChatModel()).answer(question)  # type: ignore[arg-type]


def test_question_is_stripped_once_and_reused_for_search_prompt_and_response(monkeypatch: pytest.MonkeyPatch) -> None:
    question = StripCountingString("  ¿Qué política aplica?  ")
    vector_store = FakeVectorStore(_results())
    captured: dict[str, str] = {}

    def fake_messages(context: str, question: str) -> tuple[SystemMessage, HumanMessage]:
        captured["question"] = question
        return SystemMessage(content="sistema"), HumanMessage(content="usuario")

    monkeypatch.setattr("src.rag.service.build_rag_messages", fake_messages)
    response = RagService(vector_store, FakeChatModel()).answer(question)

    assert question.strip_calls == 1
    assert vector_store.calls == [("¿Qué política aplica?", 4, 0.3)]
    assert captured["question"] == "¿Qué política aplica?"
    assert response.question == "¿Qué política aplica?"


def test_fallback_skips_context_prompt_and_model(monkeypatch: pytest.MonkeyPatch) -> None:
    vector_store = FakeVectorStore([])
    model = FakeChatModel()
    monkeypatch.setattr("src.rag.service.build_context", lambda results: pytest.fail("No debe construir contexto"))
    monkeypatch.setattr("src.rag.service.build_rag_messages", lambda **kwargs: pytest.fail("No debe construir mensajes"))

    response = RagService(vector_store, model).answer("  consulta sin evidencia  ")

    assert vector_store.calls == [("consulta sin evidencia", 4, 0.3)]
    assert model.calls == []
    assert response == RagResponse("consulta sin evidencia", FALLBACK_MESSAGE, (), 0, True)


@pytest.mark.parametrize("invalid_results", [None, (), (item for item in _results()), "texto", {"resultado": 1}])
def test_invalid_retrieval_results_raise_domain_error_without_fallback(invalid_results: object) -> None:
    model = FakeChatModel()
    vector_store = FakeVectorStore(invalid_results)

    with pytest.raises(RagRetrievalError) as error:
        RagService(vector_store, model).answer("consulta")

    assert error.value.__cause__ is None
    assert model.calls == []
    assert vector_store.calls == [("consulta", 4, 0.3)]


def test_vector_store_error_propagates_without_fallback() -> None:
    original = VectorStoreError("error de índice")
    model = FakeChatModel()

    with pytest.raises(VectorStoreError) as error:
        RagService(FakeVectorStore([], error=original), model).answer("consulta")

    assert error.value is original
    assert model.calls == []


def test_unexpected_retrieval_error_is_wrapped_without_leaking_details() -> None:
    original = RuntimeError("detalle interno de recuperación")
    model = FakeChatModel()

    with pytest.raises(RagRetrievalError) as error:
        RagService(FakeVectorStore([], error=original), model).answer("consulta")

    assert error.value.__cause__ is original
    assert "detalle interno" not in str(error.value)
    assert model.calls == []


def test_context_and_prompt_errors_propagate_without_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    context_error = RagContextError("contexto inválido")
    monkeypatch.setattr("src.rag.service.build_context", lambda results: (_ for _ in ()).throw(context_error))

    with pytest.raises(RagContextError) as raised_context:
        RagService(FakeVectorStore(_results()), FakeChatModel()).answer("consulta")
    assert raised_context.value is context_error

    prompt_error = RagPromptError("prompt inválido")
    monkeypatch.setattr("src.rag.service.build_context", build_context)
    monkeypatch.setattr("src.rag.service.build_rag_messages", lambda **kwargs: (_ for _ in ()).throw(prompt_error))

    with pytest.raises(RagPromptError) as raised_prompt:
        RagService(FakeVectorStore(_results()), FakeChatModel()).answer("consulta")
    assert raised_prompt.value is prompt_error


def test_successful_answer_reuses_components_preserves_order_and_does_not_mutate() -> None:
    results = _results()
    original_contents = [document.page_content for document, _ in results]
    original_metadata = [dict(document.metadata) for document, _ in results]
    vector_store = FakeVectorStore(results)
    model = FakeChatModel(SimpleNamespace(content=[{"type": "text", "text": "Respuesta "}, {"type": "text", "text": "con ñ"}]))

    response = RagService(vector_store, model, top_k=2, score_threshold=0.5).answer("  ¿Cuál es la política?  ")

    assert vector_store.calls == [("¿Cuál es la política?", 2, 0.5)]
    assert response.question == "¿Cuál es la política?"
    assert response.answer == "Respuesta con ñ"
    assert [source.file_name for source in response.sources] == ["politica.md", "dos.md"]
    assert response.documents_found == len(response.sources) == 2
    assert response.used_fallback is False
    assert len(model.calls) == 1
    assert isinstance(model.calls[0][0], SystemMessage)
    assert isinstance(model.calls[0][1], HumanMessage)
    assert "C| [INICIO FUENTE 1]" in model.calls[0][1].content
    assert "Q| ¿Cuál es la política?" in model.calls[0][1].content
    assert [document.page_content for document, _ in results] == original_contents
    assert [document.metadata for document, _ in results] == original_metadata


def test_existing_generation_error_is_propagated_without_double_wrapping() -> None:
    original = RagGenerationError("respuesta inválida")

    with pytest.raises(RagGenerationError) as error:
        RagService(FakeVectorStore(_results()), FakeChatModel(error=original)).answer("consulta")

    assert error.value is original
    assert error.value.__cause__ is None


def test_provider_error_is_generation_error_with_cause_and_no_fallback() -> None:
    original = RuntimeError("detalle interno del proveedor")

    with pytest.raises(RagGenerationError) as error:
        RagService(FakeVectorStore(_results()), FakeChatModel(error=original)).answer("consulta")

    assert error.value.__cause__ is original
    assert "detalle interno" not in str(error.value)


def test_empty_model_response_propagates_generation_error_without_fallback() -> None:
    model = FakeChatModel(SimpleNamespace(content=[]))

    with pytest.raises(RagGenerationError):
        RagService(FakeVectorStore(_results()), model).answer("consulta")
    assert len(model.calls) == 1


def test_response_is_immutable_and_deterministic() -> None:
    service = RagService(FakeVectorStore([]), FakeChatModel(), fallback_message="Fallback Unicode ñ")

    first = service.answer("pregunta")
    second = service.answer("pregunta")

    assert first == second
    with pytest.raises(FrozenInstanceError):
        first.answer = "otra"  # type: ignore[misc]


def test_public_exports_from_previous_blocks_remain_available() -> None:
    assert callable(build_context)
    assert callable(build_rag_messages)
    assert callable(create_chat_model)
    assert hasattr(FaissVectorStore, "from_documents")
    assert issubclass(RagRetrievalError, Exception)
