"""Pruebas focalizadas del runtime RAG reutilizable sin proveedores reales."""

from __future__ import annotations

import importlib
import os
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import dotenv
import pytest
from langchain_core.documents import Document

from src import rag
from src import app_runtime as runtime


def _environment(documents_dir: Path, vector_store_dir: Path) -> dict[str, str]:
    return {
        "GOOGLE_API_KEY": "secreto-falso-no-visible",
        "GEMINI_CHAT_MODEL": " chat-test ",
        "GEMINI_EMBEDDING_MODEL": " embedding-test ",
        "DOCUMENTS_DIR": str(documents_dir),
        "VECTOR_STORE_DIR": str(vector_store_dir),
        "RETRIEVER_TOP_K": "2",
        "RETRIEVER_SCORE_THRESHOLD": "0.4",
        "CHUNK_SIZE": "120",
        "CHUNK_OVERLAP": "20",
    }


def _config(tmp_path: Path, store_dir: Path | None = None) -> runtime.AppConfig:
    documents = tmp_path / "documents"
    documents.mkdir(exist_ok=True)
    return runtime.AppConfig(documents, store_dir or tmp_path / "store", "chat-test", "embedding-test", 2, 0.4, 120, 20)


def _write_artifacts(directory: Path, names: tuple[str, ...] = ("index.faiss", "documents.json", "manifest.json")) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).write_text("placeholder", encoding="utf-8")


def _chunk(source: str = "documento.pdf", **metadata: object) -> Document:
    return Document(page_content="contenido", metadata={"source": source, **metadata})


def test_import_from_scratch_has_no_side_effects(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original = sys.modules.pop("src.app_runtime", None)
    calls: list[str] = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: calls.append("dotenv"))
    monkeypatch.setattr(rag, "create_embeddings", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("embeddings")))
    monkeypatch.setattr(rag, "create_chat_model", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("chat")))
    try:
        importlib.import_module("src.app_runtime")
        assert calls == []
        assert list(tmp_path.iterdir()) == []
    finally:
        sys.modules.pop("src.app_runtime", None)
        if original is not None:
            sys.modules["src.app_runtime"] = original


def test_load_config_uses_dotenv_without_override_and_normalizes_models(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, bool]] = []
    documents, store = tmp_path / "documents", tmp_path / "store"
    documents.mkdir()
    monkeypatch.setattr(runtime, "load_dotenv", lambda **kwargs: calls.append(kwargs))

    config = runtime.load_app_config(_environment(documents, store))

    assert calls == [{"override": False}]
    assert (config.chat_model, config.embedding_model) == ("chat-test", "embedding-test")
    assert (config.top_k, config.score_threshold) == (2, 0.4)
    assert "secreto-falso" not in repr(config)
    with pytest.raises(FrozenInstanceError):
        config.top_k = 1  # type: ignore[misc]


def test_load_config_uses_retrieval_defaults_when_variables_are_absent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    documents, store = tmp_path / "documents", tmp_path / "store"
    documents.mkdir()
    environment = _environment(documents, store)
    environment.pop("RETRIEVER_TOP_K")
    environment.pop("RETRIEVER_SCORE_THRESHOLD")
    monkeypatch.setattr(runtime, "load_dotenv", lambda **kwargs: False)

    config = runtime.load_app_config(environment)

    assert (config.top_k, config.score_threshold) == (8, 0.55)


def test_env_example_uses_the_retrieval_defaults() -> None:
    lines = (Path(__file__).resolve().parents[1] / ".env.example").read_text(encoding="utf-8").splitlines()

    assert lines.count("RETRIEVER_TOP_K=8") == 1
    assert lines.count("RETRIEVER_SCORE_THRESHOLD=0.55") == 1


@pytest.mark.parametrize("name, value", [("GOOGLE_API_KEY", None), ("GEMINI_CHAT_MODEL", ""), ("GEMINI_EMBEDDING_MODEL", "   ")])
def test_missing_required_values_are_safe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, value: str | None) -> None:
    environment = _environment(tmp_path, tmp_path / "store")
    if value is None:
        environment.pop(name)
    else:
        environment[name] = value
    monkeypatch.setattr(runtime, "load_dotenv", lambda **kwargs: False)

    with pytest.raises(runtime.AppConfigurationError) as error:
        runtime.load_app_config(environment)
    assert name in str(error.value)
    assert "secreto-falso" not in str(error.value)


@pytest.mark.parametrize("field,value", [("RETRIEVER_TOP_K", "0"), ("RETRIEVER_TOP_K", "2x"), ("RETRIEVER_SCORE_THRESHOLD", "nan"), ("RETRIEVER_SCORE_THRESHOLD", "inf"), ("RETRIEVER_SCORE_THRESHOLD", "-inf"), ("CHUNK_SIZE", "0"), ("CHUNK_OVERLAP", "120")])
def test_invalid_numeric_config_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, value: str) -> None:
    environment = _environment(tmp_path, tmp_path / "store")
    environment[field] = value
    monkeypatch.setattr(runtime, "load_dotenv", lambda **kwargs: False)
    with pytest.raises(runtime.AppConfigurationError):
        runtime.load_app_config(environment)


def test_relative_paths_are_rooted_and_process_environment_keeps_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "repository"
    (root / "documents").mkdir(parents=True)
    process_documents = root / "process-documents"
    process_documents.mkdir()
    monkeypatch.setattr(runtime, "REPOSITORY_ROOT", root)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runtime, "load_dotenv", lambda **kwargs: False)
    environment = _environment(process_documents, root / "cache")
    environment["DOCUMENTS_DIR"] = "documents"

    config = runtime.load_app_config(environment)

    assert config.documents_dir == (root / "documents").resolve()
    assert config.vector_store_dir == (root / "cache").resolve()


@pytest.mark.parametrize("relationship", ["equal", "store_inside", "documents_inside", "dotdot", "store_file"])
def test_overlapping_or_invalid_store_paths_are_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, relationship: str) -> None:
    documents = tmp_path / "documents"
    documents.mkdir()
    if relationship == "equal":
        store = documents
    elif relationship == "store_inside":
        store = documents / "index"
    elif relationship == "documents_inside":
        store = tmp_path / "store"
        store.mkdir()
        documents.rmdir()
        documents = store / "documents"
        documents.mkdir()
    elif relationship == "dotdot":
        store = documents / "nested" / ".."
    else:
        store = tmp_path / "store-file"
        store.write_text("x", encoding="utf-8")
    environment = _environment(documents, store)
    monkeypatch.setattr(runtime, "load_dotenv", lambda **kwargs: False)

    with pytest.raises(runtime.AppConfigurationError) as error:
        runtime.load_app_config(environment)
    assert str(documents) not in str(error.value)
    assert str(store) not in str(error.value)


def test_independent_paths_are_valid_and_not_created(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    documents, store = tmp_path / "documents", tmp_path / "store"
    documents.mkdir()
    monkeypatch.setattr(runtime, "load_dotenv", lambda **kwargs: False)
    config = runtime.load_app_config(_environment(documents, store))
    assert config.vector_store_dir == store.resolve()
    assert not store.exists()


def test_count_document_units_uses_source_page_slide_sheet_and_ignores_chunk_fields() -> None:
    documents = [
        _chunk(page=1, chunk_index=1, start_index=0),
        _chunk(page=1, chunk_index=2, start_index=100),
        _chunk(page=2, chunk_index=1),
        _chunk("other.pdf", page=1),
        _chunk("sheet.xlsx", sheet="Datos", chunk_index=1),
        _chunk("sheet.xlsx", sheet="Datos", chunk_index=2),
    ]
    assert runtime._count_document_units(documents) == 4


@pytest.mark.parametrize("document", [_chunk(source=""), _chunk(source=" "), Document(page_content="x", metadata={}), _chunk(page=0), _chunk(sheet=" ")])
def test_count_document_units_rejects_insufficient_metadata(document: Document) -> None:
    with pytest.raises(runtime.AppInitializationError):
        runtime._count_document_units([document])


def test_existing_index_uses_precise_metrics_without_reindexing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(tmp_path)
    _write_artifacts(config.vector_store_dir)
    embeddings, chat = object(), object()
    chunks = [_chunk(page=1, chunk_index=1), _chunk(page=1, chunk_index=2), _chunk(page=2), _chunk("other.pdf", page=1)]
    store = SimpleNamespace(documents=chunks, index=SimpleNamespace(ntotal=4), vector_dimension=3, embedding_model="embedding-test")
    calls: list[object] = []
    service_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    answer_calls: list[object] = []
    service = SimpleNamespace(answer=lambda *args, **kwargs: answer_calls.append((args, kwargs)))
    monkeypatch.setattr(runtime, "create_embeddings", lambda model: calls.append(("embeddings", model)) or embeddings)
    monkeypatch.setattr(runtime, "create_chat_model", lambda model: calls.append(("chat", model)) or chat)
    monkeypatch.setattr(runtime, "index_directory", lambda *args, **kwargs: pytest.fail("No debe reconstruir"))
    monkeypatch.setattr(runtime.FaissVectorStore, "load", lambda directory, received: calls.append(("load", directory, received)) or store)
    monkeypatch.setattr(
        runtime,
        "RagService",
        lambda *args, **kwargs: service_calls.append((args, kwargs)) or service,
    )

    resources = runtime.initialize_app_resources(config)

    assert (resources.documents_loaded, resources.chunks_created, resources.vector_dimension, resources.index_rebuilt) == (3, 4, 3, False)
    assert calls[0] == ("embeddings", "embedding-test")
    assert calls[1] == ("chat", "chat-test")
    assert calls[2][2] is embeddings  # type: ignore[index]
    assert service_calls == [((store, chat), {"top_k": config.top_k, "score_threshold": config.score_threshold})]
    assert resources.rag_service is service
    assert answer_calls == []
    assert type(resources.index_rebuilt) is bool


@pytest.mark.parametrize("artifact", ["index.faiss", "documents.json", "manifest.json"])
@pytest.mark.parametrize("kind", ["directory", "valid_symlink"])
@pytest.mark.parametrize("force_reindex", [False, True])
def test_unsafe_artifacts_fail_before_any_rebuild_or_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact: str,
    kind: str,
    force_reindex: bool,
) -> None:
    config = _config(tmp_path)
    config.vector_store_dir.mkdir()
    if kind == "directory":
        (config.vector_store_dir / artifact).mkdir()
    else:
        target = tmp_path / "target"
        target.write_text("x", encoding="utf-8")
        try:
            (config.vector_store_dir / artifact).symlink_to(target)
        except OSError:
            pytest.skip("El sistema no permite enlaces simbólicos")
    monkeypatch.setattr(runtime, "create_embeddings", lambda model: object())
    monkeypatch.setattr(runtime, "create_chat_model", lambda model: object())
    monkeypatch.setattr(runtime, "index_directory", lambda *args, **kwargs: pytest.fail("No debe reconstruir"))
    monkeypatch.setattr(runtime.FaissVectorStore, "load", lambda *args, **kwargs: pytest.fail("No debe cargar"))

    with pytest.raises(runtime.AppInitializationError) as error:
        runtime.initialize_app_resources(config, force_reindex=force_reindex)
    assert artifact not in str(error.value)


def test_broken_symlink_is_rejected_before_rebuild(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(tmp_path)
    config.vector_store_dir.mkdir()
    try:
        (config.vector_store_dir / "index.faiss").symlink_to(tmp_path / "missing-target")
    except OSError:
        pytest.skip("El sistema no permite enlaces simbólicos")
    monkeypatch.setattr(runtime, "create_embeddings", lambda model: object())
    monkeypatch.setattr(runtime, "create_chat_model", lambda model: object())
    monkeypatch.setattr(runtime, "index_directory", lambda *args, **kwargs: pytest.fail("No debe reconstruir"))
    monkeypatch.setattr(runtime.FaissVectorStore, "load", lambda *args, **kwargs: pytest.fail("No debe cargar"))

    with pytest.raises(runtime.AppInitializationError) as error:
        runtime.initialize_app_resources(config, force_reindex=True)
    assert "missing-target" not in str(error.value)


@pytest.mark.parametrize("artifact", ["index.faiss", "documents.json", "manifest.json"])
def test_force_reindex_rejects_unsafe_artifacts_before_indexing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact: str,
) -> None:
    config = _config(tmp_path)
    config.vector_store_dir.mkdir()
    (config.vector_store_dir / artifact).mkdir()
    calls: list[str] = []
    monkeypatch.setattr(runtime, "create_embeddings", lambda model: calls.append("embeddings") or object())
    monkeypatch.setattr(runtime, "create_chat_model", lambda model: calls.append("chat") or object())
    monkeypatch.setattr(runtime, "index_directory", lambda *args, **kwargs: calls.append("index"))
    monkeypatch.setattr(runtime.FaissVectorStore, "load", lambda *args, **kwargs: calls.append("load"))

    with pytest.raises(runtime.AppInitializationError) as error:
        runtime.initialize_app_resources(config, force_reindex=True)

    assert calls == []
    assert artifact not in str(error.value)


class _TruthyObject:
    def __bool__(self) -> bool:
        return True


@pytest.mark.parametrize("force_reindex", ["sí", "false", 1, 0, None, [], _TruthyObject()])
def test_force_reindex_requires_a_real_bool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, force_reindex: object) -> None:
    config = _config(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(runtime, "_inspect_index_artifacts", lambda directory: calls.append("inspect") or False)
    monkeypatch.setattr(runtime, "create_embeddings", lambda model: calls.append("embeddings") or object())
    monkeypatch.setattr(runtime, "create_chat_model", lambda model: calls.append("chat") or object())
    monkeypatch.setattr(runtime, "index_directory", lambda *args, **kwargs: calls.append("index"))
    monkeypatch.setattr(runtime, "RagService", lambda *args, **kwargs: calls.append("service"))

    with pytest.raises(runtime.AppConfigurationError) as error:
        runtime.initialize_app_resources(config, force_reindex=force_reindex)  # type: ignore[arg-type]

    assert calls == []
    assert "sí" not in str(error.value)


@pytest.mark.parametrize("force_reindex", [False, True])
def test_force_reindex_real_bools_build_resources_with_exact_service_arguments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    force_reindex: bool,
) -> None:
    config = _config(tmp_path)
    if not force_reindex:
        _write_artifacts(config.vector_store_dir)
    embeddings, chat, service = object(), object(), object()
    store = SimpleNamespace(documents=[_chunk(page=1)], index=SimpleNamespace(ntotal=1), vector_dimension=3, embedding_model="embedding-test")
    factory_calls: list[tuple[str, object]] = []
    service_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    answer_calls: list[object] = []
    monkeypatch.setattr(runtime, "create_embeddings", lambda model: factory_calls.append(("embeddings", model)) or embeddings)
    monkeypatch.setattr(runtime, "create_chat_model", lambda model: factory_calls.append(("chat", model)) or chat)
    monkeypatch.setattr(runtime, "index_directory", lambda *args, **kwargs: factory_calls.append(("index", args[2])) or SimpleNamespace(documents_loaded=1, chunks_created=1, vector_dimension=3))
    monkeypatch.setattr(runtime.FaissVectorStore, "load", lambda directory, received: factory_calls.append(("load", received)) or store)
    monkeypatch.setattr(runtime, "RagService", lambda *args, **kwargs: service_calls.append((args, kwargs)) or service)

    resources = runtime.initialize_app_resources(config, force_reindex=force_reindex)

    expected_factory_calls = [
        ("embeddings", config.embedding_model),
        ("chat", config.chat_model),
    ]
    if force_reindex:
        expected_factory_calls.extend([("index", embeddings), ("load", embeddings)])
    else:
        expected_factory_calls.append(("load", embeddings))
    assert factory_calls == expected_factory_calls
    assert service_calls == [((store, chat), {"top_k": config.top_k, "score_threshold": config.score_threshold})]
    assert resources.rag_service is service
    assert answer_calls == []
    assert resources.index_rebuilt is force_reindex
    assert type(resources.index_rebuilt) is bool


@pytest.mark.parametrize("names", [("index.faiss",), ("index.faiss", "documents.json")])
def test_partial_regular_artifacts_rebuild(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, names: tuple[str, ...]) -> None:
    config = _config(tmp_path)
    _write_artifacts(config.vector_store_dir, names)
    _mock_rebuild_dependencies(monkeypatch, "embedding-test")
    resources = runtime.initialize_app_resources(config)
    assert resources.index_rebuilt is True


def test_changed_embedding_model_rebuilds_once_and_reloads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(tmp_path)
    _write_artifacts(config.vector_store_dir)
    calls: list[str] = []
    old = SimpleNamespace(documents=[_chunk(page=1)], index=SimpleNamespace(ntotal=1), vector_dimension=3, embedding_model="old-model")
    new = SimpleNamespace(documents=[_chunk(page=1)], index=SimpleNamespace(ntotal=1), vector_dimension=3, embedding_model="embedding-test")
    embeddings = object()
    monkeypatch.setattr(runtime, "create_embeddings", lambda model: embeddings)
    monkeypatch.setattr(runtime, "create_chat_model", lambda model: object())
    monkeypatch.setattr(runtime, "index_directory", lambda *args, **kwargs: calls.append("index") or SimpleNamespace(documents_loaded=1, chunks_created=1, vector_dimension=3))
    monkeypatch.setattr(runtime.FaissVectorStore, "load", lambda directory, received: calls.append("load") or (old if calls.count("load") == 1 else new))
    monkeypatch.setattr(runtime, "RagService", lambda *args, **kwargs: object())

    resources = runtime.initialize_app_resources(config)
    assert resources.index_rebuilt is True
    assert calls == ["load", "index", "load"]


def test_rebuilt_index_with_wrong_model_fails_safely(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(tmp_path)
    _mock_rebuild_dependencies(monkeypatch, "wrong-model")
    with pytest.raises(runtime.AppInitializationError) as error:
        runtime.initialize_app_resources(config)
    assert "wrong-model" not in str(error.value)


def _mock_rebuild_dependencies(monkeypatch: pytest.MonkeyPatch, loaded_model: str) -> None:
    store = SimpleNamespace(documents=[_chunk(page=1)], index=SimpleNamespace(ntotal=1), vector_dimension=3, embedding_model=loaded_model)
    monkeypatch.setattr(runtime, "create_embeddings", lambda model: object())
    monkeypatch.setattr(runtime, "create_chat_model", lambda model: object())
    monkeypatch.setattr(runtime, "index_directory", lambda *args, **kwargs: SimpleNamespace(documents_loaded=1, chunks_created=1, vector_dimension=3))
    monkeypatch.setattr(runtime.FaissVectorStore, "load", lambda *args, **kwargs: store)
    monkeypatch.setattr(runtime, "RagService", lambda *args, **kwargs: object())


def test_corrupt_complete_index_fails_without_rebuild(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(tmp_path)
    _write_artifacts(config.vector_store_dir)
    monkeypatch.setattr(runtime, "create_embeddings", lambda model: object())
    monkeypatch.setattr(runtime, "create_chat_model", lambda model: object())
    monkeypatch.setattr(runtime, "index_directory", lambda *args, **kwargs: pytest.fail("No debe reconstruir"))
    monkeypatch.setattr(runtime.FaissVectorStore, "load", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("índice corrupto interno")))
    with pytest.raises(runtime.AppInitializationError) as error:
        runtime.initialize_app_resources(config)
    assert "corrupto" not in str(error.value)


@pytest.mark.parametrize("stage", ["embeddings", "chat", "index", "load", "service", "metrics"])
@pytest.mark.parametrize("critical", [MemoryError, KeyboardInterrupt, SystemExit])
def test_critical_errors_propagate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str, critical: type[BaseException]) -> None:
    config = _config(tmp_path)
    _write_artifacts(config.vector_store_dir)
    store = SimpleNamespace(documents=[_chunk(page=1)], index=SimpleNamespace(ntotal=1), vector_dimension=3, embedding_model="embedding-test")
    monkeypatch.setattr(runtime, "create_embeddings", lambda model: (_ for _ in ()).throw(critical()) if stage == "embeddings" else object())
    monkeypatch.setattr(runtime, "create_chat_model", lambda model: (_ for _ in ()).throw(critical()) if stage == "chat" else object())
    monkeypatch.setattr(runtime, "index_directory", lambda *args, **kwargs: (_ for _ in ()).throw(critical()) if stage == "index" else SimpleNamespace(documents_loaded=1, chunks_created=1, vector_dimension=3))
    monkeypatch.setattr(runtime.FaissVectorStore, "load", lambda *args, **kwargs: (_ for _ in ()).throw(critical()) if stage == "load" else store)
    monkeypatch.setattr(runtime, "RagService", lambda *args, **kwargs: (_ for _ in ()).throw(critical()) if stage == "service" else object())
    if stage == "metrics":
        monkeypatch.setattr(runtime, "_count_document_units", lambda documents: (_ for _ in ()).throw(critical()))
    with pytest.raises(critical):
        runtime.initialize_app_resources(config, force_reindex=stage == "index")


def test_resources_have_only_summary_fields_and_answer_is_never_called() -> None:
    assert set(runtime.AppResources.__dataclass_fields__) == {"rag_service", "documents_loaded", "chunks_created", "vector_dimension", "index_rebuilt"}
    assert "GOOGLE_API_KEY" not in runtime.AppResources.__dataclass_fields__
