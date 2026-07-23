"""Pruebas deterministas de la interfaz Streamlit sin proveedores reales."""

from __future__ import annotations

import importlib
import inspect
import sys
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import pytest

import app
from src.app_runtime import AppConfigurationError, AppInitializationError
from src.rag.context import SourceReference


class StopCalled(BaseException):
    """Representa la interrupción real provocada por st.stop()."""


class RerunCalled(BaseException):
    """Representa la interrupción real provocada por st.rerun()."""


class _FakeStreamlit:
    def __init__(self) -> None:
        self.session_state: dict[str, Any] = {}
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.button_values: dict[str, bool] = {}
        self.checkbox_values: dict[str, bool] = {}
        self.chat_input_value: object | None = None
        self.pills_value: object | None = None
        self.sidebar = nullcontext()

    def _record(self, name: str, *args: Any, **kwargs: Any) -> None:
        self.calls.append((name, args, kwargs))

    def set_page_config(self, **kwargs: Any) -> None:
        self._record("set_page_config", **kwargs)

    def spinner(self, *args: Any, **kwargs: Any) -> Any:
        self._record("spinner", *args, **kwargs)
        return nullcontext()

    def chat_message(self, *args: Any, **kwargs: Any) -> Any:
        self._record("chat_message", *args, **kwargs)
        return nullcontext()

    def expander(self, *args: Any, **kwargs: Any) -> Any:
        self._record("expander", *args, **kwargs)
        return nullcontext()

    def button(self, label: str, **kwargs: Any) -> bool:
        self._record("button", label, **kwargs)
        return self.button_values.get(label, False)

    def checkbox(self, label: str, **kwargs: Any) -> bool:
        self._record("checkbox", label, **kwargs)
        return self.checkbox_values.get(label, False)

    def chat_input(self, *args: Any, **kwargs: Any) -> object | None:
        self._record("chat_input", *args, **kwargs)
        return self.chat_input_value

    def pills(self, *args: Any, **kwargs: Any) -> object | None:
        self._record("pills", *args, **kwargs)
        return self.pills_value

    def rerun(self) -> None:
        self._record("rerun")
        raise RerunCalled()

    def stop(self) -> None:
        self._record("stop")
        raise StopCalled()

    def __getattr__(self, name: str) -> Any:
        def record(*args: Any, **kwargs: Any) -> None:
            self._record(name, *args, **kwargs)

        return record


@pytest.fixture
def fake_st(monkeypatch: pytest.MonkeyPatch) -> _FakeStreamlit:
    fake = _FakeStreamlit()
    monkeypatch.setattr(app, "st", fake)
    return fake


def _resources(answer: object | None = None) -> SimpleNamespace:
    service = SimpleNamespace(answer=answer or (lambda question: None))
    return SimpleNamespace(
        rag_service=service,
        documents_loaded=2,
        chunks_created=5,
        vector_dimension=3,
        index_rebuilt=False,
    )


def _source(**changes: object) -> SourceReference:
    values: dict[str, object] = {
        "reference": "Fuente 1",
        "source": "politicas/reembolsos.pdf",
        "file_name": None,
        "file_type": "pdf",
        "score": 0.6844,
        "page": 1,
        "slide": None,
        "sheet": None,
        "chunk_index": 7,
    }
    values.update(changes)
    return SourceReference(**values)  # type: ignore[arg-type]


def test_import_does_not_execute_main_or_initialize_resources(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.app_runtime as runtime

    original = sys.modules.pop("app", None)
    monkeypatch.setattr(runtime, "load_app_config", lambda: pytest.fail("No debe cargar configuración"))
    monkeypatch.setattr(runtime, "initialize_app_resources", lambda *args, **kwargs: pytest.fail("No debe inicializar"))
    try:
        imported = importlib.import_module("app")
        assert hasattr(imported, "main")
    finally:
        sys.modules.pop("app", None)
        if original is not None:
            sys.modules["app"] = original


def test_configure_page_uses_required_values(fake_st: _FakeStreamlit) -> None:
    app.configure_page()
    assert fake_st.calls == [
        (
            "set_page_config",
            (),
            {
                "page_title": "Asistente de Políticas Corporativas",
                "page_icon": ":material/description:",
                "layout": "wide",
                "initial_sidebar_state": "expanded",
            },
        )
    ]


def test_session_state_initialization_preserves_existing_values(fake_st: _FakeStreamlit) -> None:
    existing_resources = object()
    existing_messages = [{"role": "user", "content": "hola"}]
    fake_st.session_state.update(app_resources=existing_resources, messages=existing_messages)
    app.initialize_session_state()
    assert fake_st.session_state["app_resources"] is existing_resources
    assert fake_st.session_state["messages"] is existing_messages


def test_load_resources_initializes_once_and_reuses_session_value(fake_st: _FakeStreamlit, monkeypatch: pytest.MonkeyPatch) -> None:
    config, resources = object(), _resources()
    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(app, "load_app_config", lambda: calls.append(("config", None)) or config)
    monkeypatch.setattr(
        app,
        "initialize_app_resources",
        lambda received, force_reindex=False: calls.append(("resources", (received, force_reindex))) or resources,
    )

    assert app.load_resources() is resources
    assert app.load_resources() is resources
    assert calls == [("config", None), ("resources", (config, False))]
    assert fake_st.session_state["app_config"] is config


@pytest.mark.parametrize("error", [AppConfigurationError("configuración pública"), AppInitializationError("inicialización pública")])
def test_load_resources_domain_errors_are_safe_and_stop(
    fake_st: _FakeStreamlit,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
    if isinstance(error, AppConfigurationError):
        monkeypatch.setattr(app, "load_app_config", lambda: (_ for _ in ()).throw(error))
    else:
        monkeypatch.setattr(app, "load_app_config", lambda: object())
        monkeypatch.setattr(app, "initialize_app_resources", lambda *args, **kwargs: (_ for _ in ()).throw(error))
    with pytest.raises(StopCalled):
        app.load_resources()
    error_messages = [args[0] for name, args, _ in fake_st.calls if name == "error"]
    assert any("pública" in message for message in error_messages)
    assert any(name == "stop" for name, _, _ in fake_st.calls)


def test_load_resources_hides_unexpected_error_details(fake_st: _FakeStreamlit, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app, "load_app_config", lambda: (_ for _ in ()).throw(RuntimeError("detalle interno")))
    with pytest.raises(StopCalled):
        app.load_resources()
    messages = [args[0] for name, args, _ in fake_st.calls if name == "error"]
    assert messages == ["No fue posible inicializar la aplicación."]


@pytest.mark.parametrize("critical", [KeyboardInterrupt, SystemExit, MemoryError])
def test_critical_initialization_errors_propagate(fake_st: _FakeStreamlit, monkeypatch: pytest.MonkeyPatch, critical: type[BaseException]) -> None:
    monkeypatch.setattr(app, "load_app_config", lambda: (_ for _ in ()).throw(critical()))
    with pytest.raises(critical):
        app.load_resources()


def test_process_question_ignores_empty_values_and_calls_answer_once(fake_st: _FakeStreamlit) -> None:
    calls: list[str] = []
    response = SimpleNamespace(answer="Respuesta final", sources=(), used_fallback=True)
    resources = _resources(answer=lambda question: calls.append(question) or response)
    app.initialize_session_state()
    app.process_question("   ", resources)
    app.process_question("  consulta válida  ", resources)
    assert calls == ["consulta válida"]
    assert fake_st.session_state["messages"] == [
        {"role": "user", "content": "consulta válida"},
        {"role": "assistant", "content": "Respuesta final", "sources": (), "used_fallback": True},
    ]


def test_process_question_does_not_store_raw_response_or_context(fake_st: _FakeStreamlit) -> None:
    source = _source()
    raw = SimpleNamespace(answer="Respuesta", sources=(source,), used_fallback=False, context="privado", prompt="privado")
    resources = _resources(answer=lambda question: raw)
    app.initialize_session_state()
    app.process_question("pregunta", resources)
    message = fake_st.session_state["messages"][-1]
    assert set(message) == {"role", "content", "sources", "used_fallback"}
    assert message["sources"] == (source,)
    assert raw not in message.values()
    assert any(name == "expander" for name, _, _ in fake_st.calls)


def test_query_error_keeps_user_message_without_false_answer(fake_st: _FakeStreamlit) -> None:
    resources = _resources(answer=lambda question: (_ for _ in ()).throw(RuntimeError("interno")))
    app.initialize_session_state()
    app.process_question("pregunta", resources)
    assert fake_st.session_state["messages"] == [{"role": "user", "content": "pregunta"}]
    error_messages = [args[0] for name, args, _ in fake_st.calls if name == "error"]
    assert error_messages == ["No fue posible procesar la consulta en este momento."]


@pytest.mark.parametrize("critical", [KeyboardInterrupt, SystemExit, MemoryError])
def test_critical_query_errors_propagate(fake_st: _FakeStreamlit, critical: type[BaseException]) -> None:
    resources = _resources(answer=lambda question: (_ for _ in ()).throw(critical()))
    app.initialize_session_state()
    with pytest.raises(critical):
        app.process_question("pregunta", resources)


def test_render_history_never_answers_again(fake_st: _FakeStreamlit) -> None:
    fake_st.session_state["messages"] = [
        {"role": "user", "content": "pregunta"},
        {"role": "assistant", "content": "respuesta", "sources": (), "used_fallback": True},
    ]
    app.render_chat_history()
    assert [args[0] for name, args, _ in fake_st.calls if name == "chat_message"] == ["user", "assistant"]


def test_header_returns_selected_suggestion_and_uses_expected_questions(fake_st: _FakeStreamlit) -> None:
    app.initialize_session_state()
    selected = app.SUGGESTED_QUESTIONS[1]
    fake_st.pills_value = selected

    assert app.render_header() == selected
    pills_calls = [entry for entry in fake_st.calls if entry[0] == "pills"]
    assert pills_calls == [
        (
            "pills",
            ("Preguntas sugeridas", app.SUGGESTED_QUESTIONS),
            {"key": "suggested_question", "width": "stretch"},
        )
    ]


def test_header_hides_suggestions_when_history_exists(fake_st: _FakeStreamlit) -> None:
    fake_st.session_state["messages"] = [{"role": "user", "content": "consulta previa"}]

    assert app.render_header() is None
    assert not any(name == "pills" for name, _, _ in fake_st.calls)


@pytest.mark.parametrize(
    ("suggested_question", "typed_question"),
    [
        (app.SUGGESTED_QUESTIONS[0], None),
        (None, app.SUGGESTED_QUESTIONS[0]),
    ],
)
def test_main_dispatches_suggested_and_typed_questions_through_the_same_flow(
    fake_st: _FakeStreamlit,
    monkeypatch: pytest.MonkeyPatch,
    suggested_question: str | None,
    typed_question: str | None,
) -> None:
    calls: list[str] = []
    response = SimpleNamespace(answer="Respuesta final", sources=(), used_fallback=True)
    resources = _resources(answer=lambda question: calls.append(question) or response)
    monkeypatch.setattr(app, "load_resources", lambda: resources)
    monkeypatch.setattr(app, "render_sidebar", lambda value: None)
    monkeypatch.setattr(app, "render_chat_history", lambda: None)
    fake_st.pills_value = suggested_question
    fake_st.chat_input_value = typed_question

    app.main()

    expected_question = app.SUGGESTED_QUESTIONS[0]
    assert calls == [expected_question]
    assert fake_st.session_state["messages"] == [
        {"role": "user", "content": expected_question},
        {"role": "assistant", "content": "Respuesta final", "sources": (), "used_fallback": True},
    ]
    assert any(name == "chat_input" for name, _, _ in fake_st.calls)


@pytest.mark.parametrize(
    ("file_name", "expected"),
    [
        ("política final.pdf", "política final.pdf"),
        ("subdirectorio/archivo con espacios.pdf", "archivo con espacios.pdf"),
        ("/interno/ruta.pdf", "Fuente no identificada"),
        (r"C:\interno\ruta.pdf", "Fuente no identificada"),
        (r"\\servidor\privado\ruta.pdf", "Fuente no identificada"),
        (None, "Fuente no identificada"),
        ("   ", "Fuente no identificada"),
        ("malicioso\narchivo.pdf", "Fuente no identificada"),
    ],
)
def test_source_reference_uses_only_safe_file_name(file_name: object, expected: str) -> None:
    line = app._format_source_reference(_source(file_name=file_name, source="/interno/secreto.pdf"))
    assert expected in line
    assert "secreto.pdf" not in line
    assert "\n" not in line


def test_render_sources_hides_controls_and_rounds_safe_score(fake_st: _FakeStreamlit) -> None:
    references = (
        _source(file_name="política.pdf", score=0.12394),
        _source(file_name="seguridad.pdf", sheet="Hoja\tprivada", score=float("nan")),
        _source(file_name="archivo.pdf", score=True),
    )
    app.render_sources(references)
    captions = [args[0] for name, args, _ in fake_st.calls if name == "caption"]
    assert any("similitud 0.124" in caption for caption in captions)
    assert all("Hoja" not in caption for caption in captions)
    assert all("\t" not in caption and "\n" not in caption for caption in captions)


def test_fallback_does_not_render_sources(fake_st: _FakeStreamlit) -> None:
    response = SimpleNamespace(answer="Fallback", sources=(_source(),), used_fallback=True)
    resources = _resources(answer=lambda question: response)
    app.initialize_session_state()
    app.process_question("pregunta", resources)
    assert not any(name == "expander" for name, _, _ in fake_st.calls)


def test_sidebar_clear_only_empties_messages_without_reinitializing(fake_st: _FakeStreamlit) -> None:
    resources = _resources()
    fake_st.session_state.update(messages=[{"role": "user", "content": "x"}], app_resources=resources)
    fake_st.button_values["Limpiar conversación"] = True
    with pytest.raises(RerunCalled):
        app.render_sidebar(resources)
    assert fake_st.session_state["messages"] == []
    assert fake_st.session_state["app_resources"] is resources
    assert any(name == "rerun" for name, _, _ in fake_st.calls)


def test_sidebar_requires_rebuild_confirmation(fake_st: _FakeStreamlit, monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[bool] = []
    monkeypatch.setattr(app, "rebuild_index", lambda: called.append(True))
    app.render_sidebar(_resources())
    rebuild_calls = [entry for entry in fake_st.calls if entry[0] == "button" and entry[1][0] == "Reconstruir índice"]
    assert rebuild_calls[0][2]["disabled"] is True
    assert called == []


def test_successful_rebuild_replaces_resources_and_clears_history(fake_st: _FakeStreamlit, monkeypatch: pytest.MonkeyPatch) -> None:
    config, old, new = object(), _resources(), _resources()
    fake_st.session_state.update(app_config=config, app_resources=old, messages=[{"role": "user", "content": "x"}])
    calls: list[tuple[object, bool]] = []
    monkeypatch.setattr(app, "initialize_app_resources", lambda received, force_reindex=False: calls.append((received, force_reindex)) or new)
    with pytest.raises(RerunCalled):
        app.rebuild_index()
    assert calls == [(config, True)]
    assert fake_st.session_state["app_resources"] is new
    assert fake_st.session_state["messages"] == []
    assert fake_st.session_state["confirm_reindex"] is False
    assert fake_st.session_state["reindex_success"] is True
    assert not any(name == "success" for name, _, _ in fake_st.calls)
    assert any(name == "rerun" for name, _, _ in fake_st.calls)


def test_failed_rebuild_preserves_prior_resources_and_history(fake_st: _FakeStreamlit, monkeypatch: pytest.MonkeyPatch) -> None:
    config, old = object(), _resources()
    history = [{"role": "user", "content": "x"}]
    fake_st.session_state.update(app_config=config, app_resources=old, messages=history)
    monkeypatch.setattr(
        app,
        "initialize_app_resources",
        lambda *args, **kwargs: (_ for _ in ()).throw(AppInitializationError("seguro")),
    )
    with pytest.raises(StopCalled):
        app.rebuild_index()
    assert fake_st.session_state["app_resources"] is old
    assert fake_st.session_state["messages"] is history
    assert not any(name == "rerun" for name, _, _ in fake_st.calls)


def test_sidebar_consumes_rebuild_success_once_after_rerun(fake_st: _FakeStreamlit) -> None:
    fake_st.session_state["reindex_success"] = True
    resources = _resources()
    app.render_sidebar(resources)
    assert [name for name, _, _ in fake_st.calls].count("success") == 1
    assert "reindex_success" not in fake_st.session_state

    fake_st.calls.clear()
    app.render_sidebar(resources)
    assert not any(name == "success" for name, _, _ in fake_st.calls)


def test_sidebar_neutralizes_invalid_metrics_and_status(fake_st: _FakeStreamlit) -> None:
    malformed = object()
    resources = SimpleNamespace(
        documents_loaded=-1,
        chunks_created=True,
        vector_dimension=malformed,
        index_rebuilt="sí",
    )
    app.render_sidebar(resources)
    metrics = [args[1] for name, args, _ in fake_st.calls if name == "metric"]
    captions = [args[0] for name, args, _ in fake_st.calls if name == "caption"]
    assert metrics == ["No disponible", "No disponible", "No disponible"]
    assert "Estado no disponible" in captions


def test_sidebar_preserves_valid_metrics_and_boolean_status(fake_st: _FakeStreamlit) -> None:
    resources = SimpleNamespace(documents_loaded=0, chunks_created=4, vector_dimension=768, index_rebuilt=True)
    app.render_sidebar(resources)
    metrics = [args[1] for name, args, _ in fake_st.calls if name == "metric"]
    captions = [args[0] for name, args, _ in fake_st.calls if name == "caption"]
    assert metrics == [0, 4, 768]
    assert "Reconstruido en esta sesión" in captions


def test_main_stops_before_rendering_when_resource_loading_stops(
    fake_st: _FakeStreamlit, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app, "load_resources", lambda: (_ for _ in ()).throw(StopCalled()))
    with pytest.raises(StopCalled):
        app.main()
    assert not any(name in {"title", "subheader", "chat_input"} for name, _, _ in fake_st.calls)


def test_main_configures_page_before_rendering(fake_st: _FakeStreamlit, monkeypatch: pytest.MonkeyPatch) -> None:
    order: list[str] = []
    resources = _resources()
    monkeypatch.setattr(app, "configure_page", lambda: order.append("page"))
    monkeypatch.setattr(app, "initialize_session_state", lambda: order.append("state"))
    monkeypatch.setattr(app, "load_resources", lambda: order.append("resources") or resources)
    monkeypatch.setattr(app, "render_header", lambda: order.append("header"))
    monkeypatch.setattr(app, "render_sidebar", lambda value: order.append("sidebar"))
    monkeypatch.setattr(app, "render_chat_history", lambda: order.append("history"))
    fake_st.chat_input_value = None
    app.main()
    assert order == ["page", "state", "resources", "header", "sidebar", "history"]


def test_source_and_security_rules_are_not_bypassed() -> None:
    source = inspect.getsource(app)
    assert "unsafe_allow_html" not in source
    assert "st.secrets" not in source
    assert "print(" not in source
    assert "create_embeddings" not in source
    assert "create_chat_model" not in source
    assert "index_directory" not in source
    assert "os.environ" not in source
    assert "dotenv" not in source
