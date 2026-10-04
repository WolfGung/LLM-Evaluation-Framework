"""HTTP helpers: auth headers, header redaction, errors that never carry the key.

Synthetic data: the key below is a made-up string and every HTTP response comes
from an httpx MockTransport. No request leaves the process.
"""

import traceback

import httpx
import pytest
from pydantic import SecretStr

from llmeval.openrouter import (
    MissingAPIKey,
    OpenRouterError,
    auth_headers,
    get_json,
    http_client,
    redact_headers,
    require_key,
    scrub,
)

FAKE_KEY = "synthetic-test-key-0123456789abcdef"


def full_text(exc: BaseException) -> str:
    """Everything a traceback would print, including chained exceptions."""
    return "".join(traceback.format_exception(exc)) + repr(exc)


def test_redact_headers_drops_credentials_in_any_case():
    headers = {
        "Authorization": f"Bearer {FAKE_KEY}",
        "X-API-Key": FAKE_KEY,
        "Cookie": "session=1",
        "set-cookie": "session=2",
        "Proxy-Authorization": "Basic abc",
        "Content-Type": "application/json",
    }

    assert redact_headers(headers) == {"Content-Type": "application/json"}


def test_redact_headers_accepts_httpx_headers():
    headers = httpx.Headers({"authorization": "Bearer x", "accept": "application/json"})

    assert redact_headers(headers) == {"accept": "application/json"}


def test_scrub_replaces_every_occurrence():
    assert scrub(f"a {FAKE_KEY} b {FAKE_KEY}", FAKE_KEY) == "a [redacted] b [redacted]"
    assert scrub("nothing here", None) == "nothing here"


def test_auth_headers_use_bearer():
    assert auth_headers(FAKE_KEY)["Authorization"] == f"Bearer {FAKE_KEY}"


def test_require_key_explains_what_is_missing():
    with pytest.raises(MissingAPIKey, match="OPENROUTER_API_KEY"):
        require_key(None)

    assert require_key(SecretStr(FAKE_KEY)) == FAKE_KEY


def test_http_error_does_not_contain_the_key():
    def echo_key(request: httpx.Request) -> httpx.Response:
        # A hostile or buggy server that echoes the credential back.
        return httpx.Response(
            401, json={"error": {"message": f"bad key {request.headers['authorization']}"}}
        )

    with (
        http_client(httpx.MockTransport(echo_key)) as client,
        pytest.raises(OpenRouterError) as caught,
    ):
        get_json(client, "/key", api_key=FAKE_KEY)

    assert caught.value.status == 401
    assert "HTTP 401" in str(caught.value)
    assert FAKE_KEY not in full_text(caught.value)
    assert caught.value.__context__ is None


def test_network_error_does_not_contain_the_key():
    def fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"refused while sending {request.headers['authorization']}")

    with (
        http_client(httpx.MockTransport(fail)) as client,
        pytest.raises(OpenRouterError) as caught,
    ):
        get_json(client, "/key", api_key=FAKE_KEY)

    assert caught.value.status is None
    assert "ConnectError" in str(caught.value)
    assert FAKE_KEY not in full_text(caught.value)
    assert caught.value.__context__ is None


def test_unencodable_key_does_not_leak():
    # httpx raises UnicodeEncodeError (not an httpx.HTTPError) while encoding
    # the header, and its repr carries the whole header value.
    bad_key = FAKE_KEY + "\u0444"
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={}))

    with http_client(transport) as client, pytest.raises(OpenRouterError) as caught:
        get_json(client, "/key", api_key=bad_key)

    assert "UnicodeEncodeError" in str(caught.value)
    assert FAKE_KEY not in full_text(caught.value)
    assert caught.value.__context__ is None


def test_non_json_body_is_an_error():
    transport = httpx.MockTransport(lambda request: httpx.Response(200, text="<html>"))

    with http_client(transport) as client, pytest.raises(OpenRouterError, match="not JSON"):
        get_json(client, "/models")
