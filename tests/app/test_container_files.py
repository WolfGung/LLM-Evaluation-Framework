"""The container files: what the image contains and how compose runs it.

Static checks of `Dockerfile`, `docker-compose.yml`, `.dockerignore` and the
package data. The image itself is built and tried by hand.
"""

import tomllib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def dockerfile_lines() -> list[str]:
    return [line.strip() for line in (ROOT / "Dockerfile").read_text().splitlines()]


def test_the_image_is_python_3_12_slim():
    assert dockerfile_lines()[0] == "FROM python:3.12-slim"


def test_the_service_runs_as_a_non_root_user():
    users = [line.split()[1] for line in dockerfile_lines() if line.startswith("USER ")]
    assert users, "no USER line"
    assert users[-1] not in ("root", "0")


def test_the_image_starts_uvicorn_with_the_app():
    # The last CMD is the image's command; an earlier one belongs to HEALTHCHECK.
    cmd = [line for line in dockerfile_lines() if line.startswith("CMD ")][-1]
    assert '"uvicorn", "app.main:app"' in cmd
    assert '"--port", "8000"' in cmd


def test_compose_runs_the_app_on_8000_in_replay_mode_by_default():
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    app = compose["services"]["app"]
    assert any(port.endswith(":8000") for port in app["ports"])
    assert "LLMEVAL_MODE=${LLMEVAL_MODE:-replay}" in app["environment"]


def test_compose_does_not_hold_a_key():
    compose = (ROOT / "docker-compose.yml").read_text()
    assert "OPENROUTER_API_KEY=${OPENROUTER_API_KEY:-}" in compose
    assert "sk-or-" not in compose


def test_the_build_context_is_an_allowlist_without_secrets():
    lines = [
        line.strip()
        for line in (ROOT / ".dockerignore").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert lines[0] == "*"
    allowed = {line[1:].rstrip("/") for line in lines if line.startswith("!")}
    assert allowed == {
        "pyproject.toml",
        "README.md",
        "LICENSE",
        "app",
        "llmeval",
        "config",
        "cassettes",
    }


def test_the_knowledge_base_and_prompts_ship_with_the_package():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    data = pyproject["tool"]["setuptools"]["package-data"]["app"]
    assert sorted(data) == ["kb/*.md", "prompts/*.md"]
