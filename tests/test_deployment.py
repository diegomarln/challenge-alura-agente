"""Comprobaciones estáticas de los artefactos de despliegue sin usar Docker."""

from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def _read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def _meaningful_lines(content: str) -> list[str]:
    return [line for line in content.splitlines() if line.strip() and not line.lstrip().startswith("#")]


def _compose_services(content: str) -> list[str]:
    lines = _meaningful_lines(content)
    services_index = next(index for index, line in enumerate(lines) if line.strip() == "services:")
    services_indent = len(lines[services_index]) - len(lines[services_index].lstrip())
    child_indent: int | None = None
    services: list[str] = []
    for line in lines[services_index + 1 :]:
        indentation = len(line) - len(line.lstrip())
        if indentation <= services_indent:
            break
        if child_indent is None:
            child_indent = indentation
        if indentation == child_indent:
            match = re.fullmatch(r"\s*([A-Za-z][A-Za-z0-9_-]*):\s*", line)
            if match:
                services.append(match.group(1))
    return services


def test_dockerfile_is_minimal_non_root_and_runs_streamlit_safely() -> None:
    dockerfile = _read("Dockerfile")

    assert "FROM python:3.13-slim" in dockerfile
    assert "COPY . ." not in dockerfile
    assert ".env" not in dockerfile
    assert dockerfile.index("COPY requirements.txt") < dockerfile.index("COPY app.py")
    for copied in ("COPY app.py", "COPY src", "COPY documents"):
        assert copied in dockerfile
    assert "requirements-dev.txt" not in dockerfile
    assert "COPY tests" not in dockerfile
    assert "apt-get install --no-install-recommends -y curl" in dockerfile
    assert "--uid 10001" in dockerfile
    assert "USER appuser" in dockerfile
    assert "EXPOSE 8501" in dockerfile
    assert "http://localhost:8501/_stcore/health" in dockerfile
    assert "--server.address=0.0.0.0" in dockerfile
    assert "--server.port=8501" in dockerfile
    assert "--server.fileWatcherType=none" in dockerfile
    assert "--browser.gatherUsageStats=false" in dockerfile
    assert "GOOGLE_API_KEY=" not in dockerfile


def test_dockerfile_ends_with_a_non_root_effective_user() -> None:
    instructions = _meaningful_lines(_read("Dockerfile"))
    users = [line.split(maxsplit=1)[1].strip() for line in instructions if re.match(r"(?i)^USER\s+", line)]

    assert users
    assert users[-1].lower() not in {"root", "0", "0:0"}
    assert all(user.lower() not in {"root", "0", "0:0"} for user in users[users.index(users[-1]) + 1 :])


def test_compose_uses_a_hardened_persistent_single_service() -> None:
    compose = _read("compose.yaml")

    assert "version:" not in compose
    assert _compose_services(compose) == ["app"]
    assert "dockerfile: Dockerfile" in compose
    assert '"8501:8501"' in compose
    assert "env_file:" in compose and "- .env" in compose
    assert "restart: unless-stopped" in compose
    assert "init: true" in compose
    assert "no-new-privileges:true" in compose
    assert "privileged:" not in compose
    assert "network_mode:" not in compose
    assert "/app/data/vector_store" in compose
    assert "vector_store_data:" in compose
    assert "http://localhost:8501/_stcore/health" in compose
    assert "GOOGLE_API_KEY=" not in compose
    assert "./:/app" not in compose


def test_dockerignore_keeps_the_build_context_free_of_local_state() -> None:
    ignored = set(_read(".dockerignore").splitlines())

    for entry in (
        ".env",
        ".env.*",
        ".venv",
        ".git",
        "tests",
        "scripts",
        "deploy",
        ".agents",
        ".claude",
        "data/vector_store",
        ".ssh/",
        "**/.ssh/",
        "id_rsa",
        "id_ed25519",
        "*.pem",
        "*.key",
        "*.ppk",
    ):
        assert entry in ignored
    for required in ("Dockerfile", "compose.yaml", "app.py", "src", "documents", "requirements.txt"):
        assert required not in ignored


def test_oci_guide_contains_safe_operational_commands_and_no_sensitive_data() -> None:
    guide = _read("deploy/oci/README.md")

    for command in (
        "docker compose up -d --build",
        "docker compose ps",
        "docker compose logs --tail=100 app",
        "docker compose logs -f app",
        "curl --fail http://localhost:8501/_stcore/health",
        "docker compose restart app",
        "docker compose down",
        "docker compose down -v",
    ):
        assert command in guide
    assert "elimina el volumen del índice vectorial" in guide
    assert "GOOGLE_API_KEY" in guide
    assert ".env" in guide
    assert "ssh-rsa" not in guide
    assert "ocid" not in guide.lower()
    assert "http://" not in guide.replace("http://localhost:8501/_stcore/health", "")
