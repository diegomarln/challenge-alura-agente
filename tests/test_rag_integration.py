"""Integración local del núcleo RAG sin servicios externos."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage, SystemMessage

from src.ingestion import load_documents
from src.rag import FALLBACK_MESSAGE, FaissVectorStore, RagResponse, RagService, split_documents


class DeterministicPolicyEmbeddings(Embeddings):
    """Vectores de tres dimensiones que separan reembolsos y teletrabajo."""

    model = "deterministic-policy-embedding-v1"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    @staticmethod
    def _vector(text: str) -> list[float]:
        normalized = text.casefold()
        reimbursement = float(any(term in normalized for term in ("reembolso", "internet")))
        remote_work = float(any(term in normalized for term in ("teletrabajo", "híbrida", "remotos", "presenciales")))
        other = 1.0 if reimbursement == 0.0 and remote_work == 0.0 else 0.1
        return [reimbursement, remote_work, other]


class FakeChatModel:
    """Modelo de chat local que registra mensajes y devuelve texto controlado."""

    def __init__(self, answer: str = "El reembolso de internet es de hasta 35 EUR mensuales.") -> None:
        self.answer = answer
        self.calls: list[tuple[object, ...]] = []

    def invoke(self, messages: tuple[object, ...]) -> object:
        self.calls.append(messages)
        return SimpleNamespace(content=self.answer)


def _write_policy_documents(tmp_path: Path) -> Path:
    documents_directory = tmp_path / "documents"
    documents_directory.mkdir()
    (documents_directory / "politica_reembolsos.md").write_text(
        "# Política de reembolsos\n\nEl reembolso de internet es de hasta 35 EUR mensuales.",
        encoding="utf-8",
    )
    (documents_directory / "politica_teletrabajo.md").write_text(
        "# Política de teletrabajo\n\nLa modalidad híbrida incluye tres días presenciales y dos remotos.",
        encoding="utf-8",
    )
    return documents_directory


def _build_store(documents_directory: Path) -> tuple[FaissVectorStore, list[object]]:
    documents = load_documents(documents_directory)
    chunks = split_documents(documents, chunk_size=300, chunk_overlap=30)
    return FaissVectorStore.from_documents(chunks, DeterministicPolicyEmbeddings()), documents


def test_rag_core_integrates_ingestion_chunking_retrieval_and_generation(tmp_path: Path) -> None:
    documents_directory = _write_policy_documents(tmp_path)
    store, loaded_documents = _build_store(documents_directory)
    chat_model = FakeChatModel()

    response = RagService(store, chat_model, top_k=2, score_threshold=0.5).answer("¿Cuál es el reembolso de internet?")

    assert len(loaded_documents) == 2
    assert store.index.ntotal == 2
    assert isinstance(response, RagResponse)
    assert response.used_fallback is False
    assert response.documents_found > 0
    assert response.documents_found == len(response.sources)
    assert response.sources[0].file_name == "politica_reembolsos.md"
    assert response.answer == "El reembolso de internet es de hasta 35 EUR mensuales."
    assert len(chat_model.calls) == 1
    assert isinstance(chat_model.calls[0][0], SystemMessage)
    assert isinstance(chat_model.calls[0][1], HumanMessage)
    assert "El reembolso de internet es de hasta 35 EUR mensuales." in chat_model.calls[0][1].content


def test_rag_core_returns_fallback_when_no_document_meets_threshold(tmp_path: Path) -> None:
    store, _ = _build_store(_write_policy_documents(tmp_path))
    chat_model = FakeChatModel()

    response = RagService(store, chat_model, top_k=2, score_threshold=0.5).answer("¿Cuál es la política de vacaciones?")

    assert response.answer == FALLBACK_MESSAGE
    assert response.sources == ()
    assert response.documents_found == 0
    assert response.used_fallback is True
    assert chat_model.calls == []


def test_persisted_store_can_be_loaded_and_used_by_rag_service(tmp_path: Path) -> None:
    store, _ = _build_store(_write_policy_documents(tmp_path))
    output_directory = tmp_path / "vector_store"
    store.save(output_directory)
    loaded_store = FaissVectorStore.load(output_directory, DeterministicPolicyEmbeddings())
    chat_model = FakeChatModel("Respuesta desde el índice cargado.")

    response = RagService(loaded_store, chat_model, top_k=2, score_threshold=0.5).answer("reembolso de internet")

    assert {path.name for path in output_directory.iterdir()} == {"index.faiss", "documents.json", "manifest.json"}
    assert not list(output_directory.glob("*.pkl"))
    assert not list(output_directory.glob("*.pickle"))
    assert response.used_fallback is False
    assert response.sources[0].file_name == "politica_reembolsos.md"
    assert response.answer == "Respuesta desde el índice cargado."
    assert len(chat_model.calls) == 1
