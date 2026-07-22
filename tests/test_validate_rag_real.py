"""Pruebas aisladas para la validación local del flujo RAG real."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from tempfile import TemporaryDirectory as RealTemporaryDirectory

import dotenv
import pytest

from scripts import validate_rag_real as script
from src import rag
from src.rag import FALLBACK_MESSAGE, RagConfigurationError, RagResponse
from src.rag.context import SourceReference


def _environment() -> dict[str, str]:
    return {
        "GOOGLE_API_KEY": "clave-falsa-que-no-debe-aparecer",
        "GEMINI_CHAT_MODEL": "chat-test",
        "GEMINI_EMBEDDING_MODEL": "embedding-test",
        "DOCUMENTS_DIR": "documents",
        "RETRIEVER_TOP_K": "2",
        "RETRIEVER_SCORE_THRESHOLD": "0.4",
        "CHUNK_SIZE": "120",
        "CHUNK_OVERLAP": "20",
    }


def _config(tmp_path: Path) -> script.RuntimeConfig:
    documents_directory = tmp_path / "documents"
    documents_directory.mkdir()
    return script.RuntimeConfig("consulta", documents_directory, 2, 0.4, 120, 20, "chat-test", "embedding-test")


def _config_with_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, dict[str, str]]:
    root = tmp_path / "repository"
    documents = root / "documents"
    documents.mkdir(parents=True)
    monkeypatch.setattr(script, "REPOSITORY_ROOT", root)
    monkeypatch.setattr(script, "load_dotenv", lambda: False)
    return root, _environment()


def test_parser_accepts_supported_arguments() -> None:
    args = script.build_parser().parse_args(
        ["--question", "consulta", "--documents-dir", "docs", "--top-k", "2", "--score-threshold", "0.5", "--chunk-size", "100", "--chunk-overlap", "10"]
    )

    assert args.question == "consulta"
    assert args.documents_dir == "docs"
    assert (args.top_k, args.score_threshold, args.chunk_size, args.chunk_overlap) == ("2", "0.5", "100", "10")


def test_relative_document_directories_are_resolved_from_repository_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, environment = _config_with_root(tmp_path, monkeypatch)
    (root / "from-environment").mkdir()
    (root / "from-cli").mkdir()
    monkeypatch.chdir(tmp_path)

    default = script.load_runtime_config(script.build_parser().parse_args([]), environment)
    environment["DOCUMENTS_DIR"] = "from-environment"
    from_environment = script.load_runtime_config(script.build_parser().parse_args([]), environment)
    from_cli = script.load_runtime_config(script.build_parser().parse_args(["--documents-dir", "from-cli"]), environment)

    assert default.documents_directory == (root / "documents").resolve()
    assert from_environment.documents_directory == (root / "from-environment").resolve()
    assert from_cli.documents_directory == (root / "from-cli").resolve()


def test_absolute_documents_directory_is_kept_and_invalid_paths_do_not_leak(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, environment = _config_with_root(tmp_path, monkeypatch)
    absolute_directory = tmp_path / "absolute-documents"
    absolute_directory.mkdir()
    environment["DOCUMENTS_DIR"] = str(absolute_directory)

    assert script.load_runtime_config(script.build_parser().parse_args([]), environment).documents_directory == absolute_directory.resolve()
    for invalid_path in (tmp_path / "missing", tmp_path / "not-a-directory.txt"):
        if invalid_path.suffix:
            invalid_path.write_text("x", encoding="utf-8")
        environment["DOCUMENTS_DIR"] = str(invalid_path)
        with pytest.raises(script.ValidationConfigurationError) as error:
            script.load_runtime_config(script.build_parser().parse_args([]), environment)
        assert str(invalid_path) not in str(error.value)


def test_load_runtime_config_uses_fake_non_secret_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    documents_directory = tmp_path / "documents"
    documents_directory.mkdir()
    environment = _environment()
    environment["DOCUMENTS_DIR"] = str(documents_directory)
    monkeypatch.setattr(script, "load_dotenv", lambda: False)

    config = script.load_runtime_config(script.build_parser().parse_args([]), environment)

    assert (config.top_k, config.score_threshold, config.chunk_size, config.chunk_overlap) == (2, 0.4, 120, 20)
    assert config.chat_model == "chat-test"
    assert config.embedding_model == "embedding-test"
    assert "clave-falsa" not in repr(config)


@pytest.mark.parametrize("name, value", [("GOOGLE_API_KEY", None), ("GEMINI_CHAT_MODEL", "   "), ("GEMINI_EMBEDDING_MODEL", "")])
def test_missing_or_blank_required_values_report_only_their_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, value: str | None) -> None:
    environment = _environment()
    environment["DOCUMENTS_DIR"] = str(tmp_path)
    if value is None:
        environment.pop(name)
    else:
        environment[name] = value
    monkeypatch.setattr(script, "load_dotenv", lambda: False)

    with pytest.raises(script.ValidationConfigurationError) as error:
        script.load_runtime_config(script.build_parser().parse_args([]), environment)

    assert name in str(error.value)
    assert "clave-falsa" not in str(error.value)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--top-k", "0"], "top-k"),
        (["--score-threshold", "nan"], "score-threshold"),
        (["--score-threshold", "inf"], "score-threshold"),
        (["--score-threshold=-inf"], "score-threshold"),
        (["--chunk-size", "20", "--chunk-overlap", "20"], "chunk-overlap"),
    ],
)
def test_invalid_cli_values_are_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, arguments: list[str], message: str) -> None:
    environment = _environment()
    environment["DOCUMENTS_DIR"] = str(tmp_path)
    monkeypatch.setattr(script, "load_dotenv", lambda: False)

    with pytest.raises(script.ValidationConfigurationError, match=message):
        script.load_runtime_config(script.build_parser().parse_args(arguments), environment)


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "invalid"])
def test_invalid_environment_threshold_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    environment = _environment()
    environment["DOCUMENTS_DIR"] = str(tmp_path)
    environment["RETRIEVER_SCORE_THRESHOLD"] = value
    monkeypatch.setattr(script, "load_dotenv", lambda: False)

    with pytest.raises(script.ValidationConfigurationError, match="score-threshold"):
        script.load_runtime_config(script.build_parser().parse_args([]), environment)


@pytest.mark.parametrize(
    "value",
    [
        "politica.pdf",
        None,
        "",
        "/secreto/politica.pdf",
        "C:\\secreto\\politica.pdf",
        "C:/secreto/politica.pdf",
        "\\\\servidor\\compartido\\politica.pdf",
        "subdirectorio/politica.pdf",
        "subdirectorio\\politica.pdf",
        "politica\ninyectada.pdf",
        "politica\rinyectada.pdf",
        "politica\tinyeccion.pdf",
        "politica\x01control.pdf",
    ],
)
def test_safe_source_name_never_leaks_paths_or_controls(value: object) -> None:
    output = script.format_source(SourceReference("Fuente 1", None, value, "pdf", 0.9, 1, None, None, 1))

    assert "\n" not in output
    assert "\r" not in output
    assert "\t" not in output
    assert "/secreto" not in output
    assert "C:\\secreto" not in output
    assert "servidor" not in output
    if value in {"politica.pdf", "subdirectorio/politica.pdf", "subdirectorio\\politica.pdf"}:
        assert "archivo=politica.pdf" in output
    else:
        assert f"archivo={script.UNKNOWN_SOURCE_NAME}" in output


def test_run_validation_coordinates_components_with_identical_embeddings_and_cleans_temp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(tmp_path)
    calls: dict[str, object] = {"counts": {}}
    embeddings = object()
    chat_model = object()
    source = SourceReference("Fuente 1", None, "politica.pdf", "pdf", 0.9, 2, None, None, 1)
    response = RagResponse("consulta", "respuesta controlada", (source,), 1, False)

    class FakeStore:
        vector_dimension = 3

    class FakeService:
        def __init__(self, store: object, model: object, top_k: int, score_threshold: float) -> None:
            calls["counts"]["service"] = calls["counts"].get("service", 0) + 1  # type: ignore[index]
            calls["service"] = (store, model, top_k, score_threshold)

        def answer(self, question: str) -> RagResponse:
            calls["counts"]["answer"] = calls["counts"].get("answer", 0) + 1  # type: ignore[index]
            calls["question"] = question
            return response

    def fake_index(documents: Path, output: Path, received_embeddings: object, **kwargs: object) -> object:
        calls["counts"]["index"] = calls["counts"].get("index", 0) + 1  # type: ignore[index]
        calls["index"] = (documents, output, received_embeddings, kwargs)
        return type("IndexingResult", (), {"documents_loaded": 2, "chunks_created": 3, "vector_dimension": 3})()

    def fake_load(directory: Path, received_embeddings: object) -> FakeStore:
        calls["counts"]["load"] = calls["counts"].get("load", 0) + 1  # type: ignore[index]
        calls["load"] = (directory, received_embeddings)
        return FakeStore()

    def fake_embeddings(model: str) -> object:
        calls["counts"]["embeddings"] = calls["counts"].get("embeddings", 0) + 1  # type: ignore[index]
        return embeddings

    def fake_chat(model: str) -> object:
        calls["counts"]["chat"] = calls["counts"].get("chat", 0) + 1  # type: ignore[index]
        return chat_model

    monkeypatch.setattr(script, "create_embeddings", fake_embeddings)
    monkeypatch.setattr(script, "create_chat_model", fake_chat)
    monkeypatch.setattr(script, "index_directory", fake_index)
    monkeypatch.setattr(script.FaissVectorStore, "load", fake_load)
    monkeypatch.setattr(script, "RagService", FakeService)

    result = script.run_validation(config)
    output = calls["index"][1]  # type: ignore[index]

    assert result.response is response
    assert calls["counts"] == {"embeddings": 1, "chat": 1, "index": 1, "load": 1, "service": 1, "answer": 1}
    assert calls["index"][2] is embeddings  # type: ignore[index]
    assert calls["load"][1] is embeddings  # type: ignore[index]
    assert calls["load"][0] == output  # type: ignore[index]
    assert isinstance(output, Path)
    assert not str(output).startswith(str(script.REPOSITORY_ROOT))
    assert not output.exists()


@pytest.mark.parametrize("stage", ["index", "load", "service", "answer"])
def test_run_validation_cleans_temporary_directory_on_stage_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str) -> None:
    config = _config(tmp_path)
    created_paths: list[Path] = []

    class FakeStore:
        vector_dimension = 3

    class FakeService:
        def __init__(self, *args: object, **kwargs: object) -> None:
            if stage == "service":
                raise RuntimeError("interno")

        def answer(self, question: str) -> RagResponse:
            if stage == "answer":
                raise RuntimeError("interno")
            return RagResponse("consulta", "respuesta", (), 0, True)

    def fake_index(documents: Path, output: Path, embeddings: object, **kwargs: object) -> object:
        created_paths.append(output)
        output.mkdir(exist_ok=True)
        (output / "index.faiss").write_text("temporal", encoding="utf-8")
        if stage == "index":
            raise RuntimeError("interno")
        return type("IndexingResult", (), {"documents_loaded": 1, "chunks_created": 1, "vector_dimension": 3})()

    def fake_load(directory: Path, embeddings: object) -> FakeStore:
        if stage == "load":
            raise RuntimeError("interno")
        return FakeStore()

    monkeypatch.setattr(script, "create_embeddings", lambda model: object())
    monkeypatch.setattr(script, "create_chat_model", lambda model: object())
    monkeypatch.setattr(script, "index_directory", fake_index)
    monkeypatch.setattr(script.FaissVectorStore, "load", fake_load)
    monkeypatch.setattr(script, "RagService", FakeService)

    with pytest.raises(RuntimeError, match="interno"):
        script.run_validation(config)

    assert created_paths
    assert not created_paths[0].exists()
    assert not list(tmp_path.rglob("index.faiss"))
    assert not (script.REPOSITORY_ROOT / "data").exists() or not list((script.REPOSITORY_ROOT / "data").glob("index.faiss"))


def test_format_summary_failure_happens_after_temporary_cleanup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(tmp_path)
    temporary_paths: list[Path] = []

    class FakeStore:
        vector_dimension = 3

    class FakeService:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def answer(self, question: str) -> RagResponse:
            return RagResponse("consulta", "respuesta", (), 0, True)

    def fake_index(documents: Path, output: Path, embeddings: object, **kwargs: object) -> object:
        temporary_paths.append(output)
        output.mkdir(exist_ok=True)
        return type("IndexingResult", (), {"documents_loaded": 1, "chunks_created": 1, "vector_dimension": 3})()

    monkeypatch.setattr(script, "load_runtime_config", lambda args: config)
    monkeypatch.setattr(script, "create_embeddings", lambda model: object())
    monkeypatch.setattr(script, "create_chat_model", lambda model: object())
    monkeypatch.setattr(script, "index_directory", fake_index)
    monkeypatch.setattr(script.FaissVectorStore, "load", lambda directory, embeddings: FakeStore())
    monkeypatch.setattr(script, "RagService", FakeService)
    monkeypatch.setattr(script, "format_summary", lambda result: (_ for _ in ()).throw(RuntimeError("interno")))

    assert script.main([]) == 1
    assert temporary_paths and not temporary_paths[0].exists()


@pytest.mark.parametrize(
    ("factory_name", "factory_error", "expected_message"),
    [
        ("create_embeddings", ValueError("detalle sensible de embeddings"), "embeddings"),
        ("create_embeddings", RuntimeError("detalle sensible inesperado"), "embeddings"),
        ("create_chat_model", RagConfigurationError("detalle sensible de chat"), "modelo de chat"),
        ("create_chat_model", RuntimeError("detalle sensible inesperado"), "modelo de chat"),
    ],
)
def test_provider_initialization_errors_are_categorized_without_details(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, factory_name: str, factory_error: Exception, expected_message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    config = _config(tmp_path)
    monkeypatch.setattr(script, "load_runtime_config", lambda args: config)
    if factory_name == "create_embeddings":
        monkeypatch.setattr(script, "create_embeddings", lambda model: (_ for _ in ()).throw(factory_error))
        monkeypatch.setattr(script, "create_chat_model", lambda model: object())
    else:
        monkeypatch.setattr(script, "create_embeddings", lambda model: object())
        monkeypatch.setattr(script, "create_chat_model", lambda model: (_ for _ in ()).throw(factory_error))

    assert script.main([]) == 1

    output = capsys.readouterr().err
    assert "Error de proveedor" in output
    assert expected_message in output
    assert "detalle sensible" not in output
    assert "Traceback" not in output


def test_keyboard_interrupt_is_not_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(script, "load_runtime_config", lambda args: (_ for _ in ()).throw(KeyboardInterrupt()))

    with pytest.raises(KeyboardInterrupt):
        script.main([])


def test_import_from_scratch_has_no_side_effects(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original_module = sys.modules.pop("scripts.validate_rag_real", None)
    calls: list[str] = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: calls.append("dotenv"))
    monkeypatch.setattr(rag, "create_embeddings", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("embeddings")))
    monkeypatch.setattr(rag, "create_chat_model", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("chat")))
    monkeypatch.setattr(rag, "index_directory", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("index")))
    try:
        importlib.import_module("scripts.validate_rag_real")
        assert calls == []
        assert list(tmp_path.iterdir()) == []
    finally:
        sys.modules.pop("scripts.validate_rag_real", None)
        if original_module is not None:
            sys.modules["scripts.validate_rag_real"] = original_module


def test_help_does_not_load_dotenv_or_initialize_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(script, "load_dotenv", lambda: (_ for _ in ()).throw(AssertionError("dotenv")))
    monkeypatch.setattr(script, "create_embeddings", lambda model: (_ for _ in ()).throw(AssertionError("embeddings")))
    monkeypatch.setattr(script, "create_chat_model", lambda model: (_ for _ in ()).throw(AssertionError("chat")))

    with pytest.raises(SystemExit) as error:
        script.main(["--help"])

    assert error.value.code == 0
