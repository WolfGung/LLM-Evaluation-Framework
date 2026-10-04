"""The HTTP service, served by uvicorn on a loopback port and called with httpx.

The model behind the service is the synthetic fake from `fakes.py` (made-up
replies), injected through `create_app(client=...)`, except in the tests of the
default replay path, which read an empty temporary cassette directory.
"""

import json

import httpx
import pytest
from pydantic import SecretStr

from app.main import ServiceConfigError, build_client, create_app
from app.retrieval import load_kb
from llmeval.client import MissingRecording, ModelClient
from llmeval.config import Config, Mode, Settings, load_models_config
from llmeval.openrouter import MissingAPIKey, OpenRouterError
from llmeval.quota import QuotaExhausted
from tests.app.fakes import STRUCTURED, UNSTRUCTURED, FakeModel
from tests.app.server import ServerDidNotStart, serve

GOOD_TRIAGE = {
    "category": "returns",
    "priority": "normal",
    "order_id": "TS-555123",
    "summary": "Customer wants to return a sander.",
}


def call(app, method, path, **kwargs):
    with serve(app) as base, httpx.Client(base_url=base, timeout=10) as http:
        return http.request(method, path, **kwargs)


def fake_app(model, role=UNSTRUCTURED):
    return create_app(client=model, role=role)


def config(mode: Mode, key: str | None = None) -> Config:
    return Config(
        models=load_models_config(),
        settings=Settings(mode=mode, api_key=SecretStr(key) if key else None),
    )


def test_health():
    response = call(fake_app(FakeModel()), "GET", "/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["documents"] == len(load_kb())
    assert body["prompt_versions"] == {"assistant": ["v1", "v2"], "triage": ["v1", "v2"]}


def test_assist_returns_the_answer_and_its_citations():
    model = FakeModel(reply="Within 30 days [kb-returns].")
    response = call(
        fake_app(model),
        "POST",
        "/assist",
        json={"question": "How long can I return a drill?", "version": "v2"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Within 30 days [kb-returns]."
    assert body["citations"] == ["kb-returns"]
    assert "kb-returns" in body["retrieved"]
    assert body["version"] == "v2"
    assert body["model"] == UNSTRUCTURED.model
    assert body["empty_reason"] is None
    assert model.calls[0]["messages"][1]["content"] == "How long can I return a drill?"


def test_assist_reports_an_empty_answer_as_a_result():
    model = FakeModel(reply="", finish_reason="length")
    response = call(fake_app(model), "POST", "/assist", json={"question": "Q?", "version": "v1"})
    assert response.status_code == 200
    assert response.json()["answer"] == ""
    assert response.json()["empty_reason"] == "length"


@pytest.mark.parametrize(
    "payload",
    [
        {"question": "Q?", "version": "v9"},
        {"question": "Q?"},
        {"question": "   ", "version": "v1"},
        {"question": "x" * 2001, "version": "v1"},
        {"question": "Q?", "version": "v1", "k": 10},
        {"version": "v1"},
    ],
)
def test_assist_refuses_bad_requests(payload):
    model = FakeModel(reply="unused")
    response = call(fake_app(model), "POST", "/assist", json=payload)
    assert response.status_code == 422
    assert model.calls == []


def test_triage_returns_the_validated_result():
    model = FakeModel(reply=json.dumps(GOOD_TRIAGE))
    response = call(
        fake_app(model),
        "POST",
        "/triage",
        json={"text": "I want to return my sander", "version": "v2"},
    )
    assert response.status_code == 200
    assert response.json() == {
        "result": GOOD_TRIAGE,
        "version": "v2",
        "model": UNSTRUCTURED.model,
    }


def test_triage_sends_the_schema_for_a_structured_role():
    model = FakeModel(reply=json.dumps(GOOD_TRIAGE))
    response = call(
        fake_app(model, STRUCTURED), "POST", "/triage", json={"text": "Return?", "version": "v1"}
    )
    assert response.status_code == 200
    assert model.calls[0]["body"]["provider"] == {"require_parameters": True}


@pytest.mark.parametrize(
    ("reply", "finish_reason", "kind"),
    [
        ("Sure! The category is returns.", "stop", "invalid_json"),
        ('{"category": "returns"}', "stop", "invalid_schema"),
        ("", "length", "empty"),
    ],
)
def test_triage_reports_an_invalid_reply_with_the_raw_text(reply, finish_reason, kind):
    model = FakeModel(reply=reply, finish_reason=finish_reason)
    response = call(fake_app(model), "POST", "/triage", json={"text": "Help", "version": "v1"})
    assert response.status_code == 502
    body = response.json()
    assert body["kind"] == kind
    assert body["raw"] == reply
    assert body["detail"].startswith("triage reply is")


@pytest.mark.parametrize(
    "payload",
    [
        {"text": "", "version": "v1"},
        {"text": "Help", "version": "v3"},
        {"text": "x" * 5001, "version": "v1"},
    ],
)
def test_triage_refuses_bad_requests(payload):
    response = call(fake_app(FakeModel()), "POST", "/triage", json=payload)
    assert response.status_code == 422


@pytest.mark.parametrize(
    ("error", "status", "detail"),
    [
        (
            MissingRecording("a" * 64, 0, None),
            503,
            "no recording for key aaaaaaaaaaaa/0: run make record",
        ),
        (QuotaExhausted(reset_at=None), 503, "free daily quota reached"),
        (OpenRouterError("POST /chat/completions failed: HTTP 500", 500), 502, "HTTP 500"),
    ],
)
@pytest.mark.parametrize(
    ("path", "payload"),
    [("/assist", {"question": "Q?", "version": "v1"}), ("/triage", {"text": "T", "version": "v1"})],
)
def test_model_failures_become_clean_http_errors(error, status, detail, path, payload):
    response = call(fake_app(FakeModel(error=error)), "POST", path, json=payload)
    assert response.status_code == status
    assert detail in response.json()["detail"]
    assert "Traceback" not in response.text


def test_the_default_client_answers_a_replay_miss_with_503(tmp_path):
    app = create_app(config=config(Mode.REPLAY), cassettes_dir=tmp_path)
    with serve(app) as base, httpx.Client(base_url=base, timeout=10) as http:
        health = http.get("/health")
        assist = http.post("/assist", json={"question": "Can I pay with PayPal?", "version": "v1"})
        triage = http.post("/triage", json={"text": "Where is TS-123456?", "version": "v2"})
    assert health.json()["mode"] == "replay"
    for response in (assist, triage):
        assert response.status_code == 503
        assert response.json()["detail"].startswith("no recording for adhoc/")
        assert response.json()["detail"].endswith(": run make record")


def test_build_client_uses_the_configured_mode_and_cassettes(tmp_path):
    client = build_client(config(Mode.REPLAY), tmp_path)
    assert isinstance(client, ModelClient)
    assert client.mode is Mode.REPLAY
    assert client.store.root == tmp_path


def test_the_service_refuses_record_mode(tmp_path):
    with pytest.raises(ServiceConfigError, match="make record"):
        build_client(config(Mode.RECORD, key="synthetic-key-123"), tmp_path)


def test_live_mode_needs_a_key(tmp_path):
    with pytest.raises(MissingAPIKey):
        build_client(config(Mode.LIVE), tmp_path)


def test_a_bad_mode_stops_the_server_at_startup(tmp_path):
    app = create_app(config=config(Mode.RECORD, key="synthetic-key-123"), cassettes_dir=tmp_path)
    with pytest.raises(ServerDidNotStart), serve(app):
        pass


def test_the_module_level_app_is_importable_without_configuration():
    from app.main import app

    assert app.title == "Toolshop"


def test_calls_are_tagged_with_the_requested_version():
    model = FakeModel(reply="ok")
    call(fake_app(model), "POST", "/assist", json={"question": "Q?", "version": "v2"})
    assert model.calls[0]["tag"].version == "v2"
