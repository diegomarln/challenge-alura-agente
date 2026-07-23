"""Interfaz Streamlit para consultar políticas corporativas con el núcleo RAG."""

from __future__ import annotations

import math
from collections.abc import Iterable
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any

import streamlit as st

from src.app_runtime import (
    AppConfigurationError,
    AppInitializationError,
    AppResources,
    initialize_app_resources,
    load_app_config,
)
from src.rag.context import SourceReference
from src.rag.generation import RagError
from src.rag.vector_store import VectorStoreError


SUGGESTED_QUESTIONS = {
    "Reembolso de internet": "¿Cuál es el monto máximo mensual que se puede reembolsar por internet?",
    "Días de teletrabajo": "¿Cuántos días de teletrabajo se permiten por semana?",
    "Correo sospechoso": "¿Qué debo hacer si recibo un correo sospechoso?",
}
FALLBACK_GUIDANCE = "Prueba con una consulta relacionada con reembolsos, teletrabajo o seguridad."


def configure_page() -> None:
    """Configura la página antes de renderizar cualquier otro elemento."""
    st.set_page_config(
        page_title="Asistente de Políticas Corporativas",
        page_icon=":material/description:",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def initialize_session_state() -> None:
    """Inicializa exclusivamente el estado mínimo persistente de la sesión."""
    st.session_state.setdefault("messages", [])


def load_resources(force_reindex: bool = False) -> AppResources | None:
    """Carga recursos una vez por sesión o reconstruye el índice bajo confirmación."""
    initialize_session_state()
    if type(force_reindex) is not bool:
        raise ValueError("force_reindex debe ser booleano")
    if not force_reindex and "app_resources" in st.session_state:
        return st.session_state["app_resources"]

    config = st.session_state.get("app_config")
    if config is None:
        try:
            config = load_app_config()
        except (KeyboardInterrupt, SystemExit, MemoryError):
            raise
        except AppConfigurationError as error:
            _show_initialization_error(error)
            return None
        except Exception:
            _show_initialization_error(None)
            return None
        st.session_state["app_config"] = config

    try:
        with st.spinner("Preparando el conocimiento interno..."):
            resources = initialize_app_resources(config, force_reindex=force_reindex)
    except (KeyboardInterrupt, SystemExit, MemoryError):
        raise
    except AppConfigurationError as error:
        _show_initialization_error(error)
        return None
    except AppInitializationError as error:
        _show_initialization_error(error)
        return None
    except Exception:
        _show_initialization_error(None)
        return None
    st.session_state["app_resources"] = resources
    return resources


def render_header() -> str | None:
    """Muestra la orientación inicial y devuelve una sugerencia seleccionada."""
    st.header("Asistente de Políticas Corporativas", divider=False)
    st.caption("Consulta políticas de reembolsos, teletrabajo y seguridad.")
    st.caption(":material/verified: Las respuestas se basan en documentos internos.")
    pending_question = st.session_state.get("pending_suggested_question")
    if st.session_state["messages"] or pending_question is not None:
        return pending_question if isinstance(pending_question, str) else None
    with st.container(border=True):
        st.write("Bienvenido. Elige una pregunta sugerida o escribe la tuya.")
        st.caption("Puedes consultar condiciones y procedimientos de las políticas internas.")
        st.pills(
            "Preguntas sugeridas",
            tuple(SUGGESTED_QUESTIONS),
            key="suggested_question",
            on_change=_queue_suggested_question,
            width="stretch",
        )
    pending_question = st.session_state.get("pending_suggested_question")
    return pending_question if isinstance(pending_question, str) else None


def _queue_suggested_question() -> None:
    """Convierte una etiqueta breve en su consulta y elimina la selección visible."""
    selected_label = st.session_state.get("suggested_question")
    question = SUGGESTED_QUESTIONS.get(selected_label) if isinstance(selected_label, str) else None
    if question is not None:
        st.session_state["pending_suggested_question"] = question
    st.session_state.pop("suggested_question", None)


def render_sidebar(resources: AppResources) -> None:
    """Presenta métricas seguras y controles explícitos de la sesión e índice."""
    with st.sidebar:
        if st.session_state.pop("reindex_success", False) is True:
            st.success("El índice se reconstruyó correctamente.")
        st.subheader("Conocimiento disponible")
        st.metric("Unidades documentales", _safe_metric_value(resources.documents_loaded))
        st.metric("Fragmentos indexados", _safe_metric_value(resources.chunks_created))
        st.metric("Dimensión vectorial", _safe_metric_value(resources.vector_dimension))
        st.caption(_index_status(resources.index_rebuilt))

        st.subheader("Conversación")
        if st.button("Limpiar conversación", key="clear_conversation"):
            st.session_state["messages"] = []
            st.rerun()

        st.divider()
        with st.expander("Administración", expanded=False):
            st.caption("Reconstruye el índice cuando cambien los documentos o la configuración de fragmentación.")
            confirmed = st.checkbox("Confirmo la reconstrucción del índice", key="confirm_reindex")
            if st.button("Reconstruir índice", disabled=not confirmed, key="rebuild_index"):
                rebuild_index()


def rebuild_index() -> None:
    """Reconstruye una única vez y conserva el estado previo si la operación falla."""
    if "app_config" not in st.session_state:
        _show_initialization_error(None)
        return
    previous_resources = st.session_state.get("app_resources")
    previous_messages = st.session_state["messages"]
    resources = load_resources(force_reindex=True)
    if resources is None:
        if previous_resources is not None:
            st.session_state["app_resources"] = previous_resources
        st.session_state["messages"] = previous_messages
        return
    st.session_state["messages"] = []
    st.session_state["confirm_reindex"] = False
    st.session_state["reindex_success"] = True
    st.rerun()


def render_chat_history() -> None:
    """Renderiza el historial seguro sin consultar nuevamente al servicio RAG."""
    for message in st.session_state["messages"]:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            continue
        with st.chat_message(role):
            st.markdown(content)
            if role == "assistant":
                _render_response_details(message.get("sources"), message.get("used_fallback"))


def process_question(question: object, resources: AppResources) -> None:
    """Envía una pregunta válida una sola vez y almacena solo el resultado seguro."""
    if not isinstance(question, str):
        return
    normalized_question = question.strip()
    if not normalized_question:
        return

    st.session_state["messages"].append({"role": "user", "content": normalized_question})
    with st.chat_message("user"):
        st.markdown(normalized_question)
    try:
        with st.chat_message("assistant"):
            with st.spinner("Consultando documentos internos..."):
                response = resources.rag_service.answer(normalized_question)
            answer, sources, used_fallback = _safe_response(response)
            if answer is None:
                st.error("No fue posible procesar la respuesta del asistente.")
                return
            st.markdown(answer)
            _render_response_details(sources, used_fallback)
    except (KeyboardInterrupt, SystemExit, MemoryError):
        raise
    except (RagError, VectorStoreError):
        st.error("No fue posible consultar los documentos internos.")
        return
    except Exception:
        st.error("No fue posible procesar la consulta en este momento.")
        return
    st.session_state["messages"].append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "used_fallback": used_fallback,
        }
    )


def render_sources(sources: Iterable[SourceReference]) -> None:
    """Muestra referencias normalizadas sin exponer rutas ni contenido documental."""
    with st.expander("Fuentes consultadas", expanded=False):
        for source in sources:
            st.caption(_format_source_reference(source))


def _render_response_details(sources: object, used_fallback: object) -> None:
    """Muestra evidencia o guía de fallback sin alterar el contenido de la respuesta."""
    if used_fallback is True:
        st.caption(FALLBACK_GUIDANCE)
        return
    if not isinstance(sources, tuple) or not sources:
        return
    st.badge(
        f"Basada en documentos internos · {len(sources)} fuentes",
        icon=":material/verified:",
        color="blue",
    )
    render_sources(sources)


def _safe_response(response: object) -> tuple[str | None, tuple[SourceReference, ...], bool]:
    """Extrae el mínimo resultado visualizable sin almacenar objetos del proveedor."""
    answer = _safe_attribute(response, "answer")
    used_fallback = _safe_attribute(response, "used_fallback")
    sources = _safe_attribute(response, "sources")
    if not isinstance(answer, str) or not answer.strip() or type(used_fallback) is not bool:
        return None, (), False
    safe_sources = sources if isinstance(sources, tuple) and all(isinstance(item, SourceReference) for item in sources) else ()
    return answer, safe_sources, used_fallback


def _format_source_reference(source: object) -> str:
    """Construye una línea de fuente sin rutas, controles ni conversiones implícitas."""
    name = _safe_file_name(_safe_attribute(source, "file_name"))
    parts = [name]
    page = _positive_int(_safe_attribute(source, "page"))
    slide = _positive_int(_safe_attribute(source, "slide"))
    sheet = _safe_label(_safe_attribute(source, "sheet"))
    chunk_index = _positive_int(_safe_attribute(source, "chunk_index"))
    score = _safe_score(_safe_attribute(source, "score"))
    if page is not None:
        parts.append(f"página {page}")
    if slide is not None:
        parts.append(f"diapositiva {slide}")
    if sheet is not None:
        parts.append(f"hoja {sheet}")
    if chunk_index is not None:
        parts.append(f"fragmento {chunk_index}")
    if score is not None:
        parts.append(f"similitud {score:.3f}")
    return " · ".join(parts)


def _safe_file_name(file_name: object) -> str:
    """Devuelve solo un basename relativo de ``file_name`` sin usar otros campos."""
    value = _safe_label(file_name)
    if value is None:
        return "Fuente no identificada"
    normalized = value.replace("\\", "/")
    if normalized.startswith("/") or PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute():
        return "Fuente no identificada"
    name = normalized.rsplit("/", 1)[-1]
    if name and name not in {".", ".."}:
        return name
    return "Fuente no identificada"


def _safe_label(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip() or any(ord(character) < 32 or ord(character) == 127 for character in value):
        return None
    return value.strip()


def _safe_attribute(value: object, attribute: str) -> object | None:
    try:
        return getattr(value, attribute)
    except Exception:
        return None


def _positive_int(value: object) -> int | None:
    return value if type(value) is int and value > 0 else None


def _safe_score(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def _safe_metric_value(value: object) -> int | str:
    """Evita enviar valores arbitrarios de sesión a componentes visuales."""
    return value if type(value) is int and value >= 0 else "No disponible"


def _index_status(value: object) -> str:
    if value is True:
        return "Reconstruido en esta sesión"
    if value is False:
        return "Cargado desde disco"
    return "Estado no disponible"


def _show_initialization_error(error: AppConfigurationError | AppInitializationError | None) -> None:
    if isinstance(error, AppConfigurationError):
        st.error(f"Error de configuración: {error}")
    elif isinstance(error, AppInitializationError):
        st.error(f"No fue posible inicializar los recursos: {error}")
    else:
        st.error("No fue posible inicializar la aplicación.")
    st.stop()


def main() -> None:
    """Ejecuta la pantalla Streamlit sin efectos durante la importación."""
    configure_page()
    initialize_session_state()
    resources = load_resources()
    if resources is None:
        return
    suggested_question = render_header()
    render_sidebar(resources)
    render_chat_history()
    typed_question = st.chat_input("Escribe una pregunta sobre las políticas internas")
    question = suggested_question if suggested_question is not None else typed_question
    if question is not None:
        process_question(question, resources)
        if suggested_question is not None:
            st.session_state.pop("pending_suggested_question", None)


if __name__ == "__main__":
    main()
