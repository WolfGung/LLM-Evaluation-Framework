"""Small HTTP layer for the OpenRouter API, built so the API key cannot leak.

Three rules live here:
- the key is read from the environment by `config.load_settings` and only
  enters a request as the `Authorization` header;
- `redact_headers` removes credentials from any header mapping before it is
  logged or stored;
- errors are rebuilt from scratch with the key scrubbed out and with no chained
  httpx exception, because an httpx exception holds the request, and the
  request holds the header.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import httpx
from pydantic import SecretStr

BASE_URL = "https://openrouter.ai/api/v1"
TIMEOUT_S = 60.0
APP_TITLE = "LLM-Evaluation-Framework"

SENSITIVE_HEADERS = frozenset(
    {"authorization", "proxy-authorization", "x-api-key", "cookie", "set-cookie"}
)
REDACTED = "[redacted]"


class MissingAPIKey(RuntimeError):
    """A live or record run was started without OPENROUTER_API_KEY."""


class OpenRouterError(RuntimeError):
    """The API answered with an error or could not be reached.

    `status` is the HTTP status, or None when no response arrived.
    """

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


def redact_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Return a copy of `headers` without credential headers (any letter case)."""
    return {name: value for name, value in headers.items() if name.lower() not in SENSITIVE_HEADERS}


def scrub(text: str, secret: str | None) -> str:
    """Replace every occurrence of `secret` in `text`, raw or escaped.

    The escaped form matters because some errors quote a header as bytes,
    for example h11's "Illegal header value b'Bearer abc\\ndef'".
    """
    if not secret:
        return text
    for form in (secret, repr(secret)[1:-1]):
        text = text.replace(form, REDACTED)
    return text


def require_key(api_key: SecretStr | None) -> str:
    if api_key is None:
        raise MissingAPIKey(
            "OPENROUTER_API_KEY is not set. Live and record modes call the API and need it; "
            "replay mode answers from the cassettes and needs no key."
        )
    return api_key.get_secret_value()


def auth_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}", "X-OpenRouter-Title": APP_TITLE}


def http_client(
    transport: httpx.BaseTransport | None = None, timeout: float = TIMEOUT_S
) -> httpx.Client:
    """An httpx client for OpenRouter; tests pass a MockTransport."""
    return httpx.Client(base_url=BASE_URL, transport=transport, timeout=timeout)


def send(
    client: httpx.Client,
    method: str,
    path: str,
    *,
    api_key: str | None = None,
    json_body: Mapping[str, Any] | None = None,
) -> httpx.Response:
    """Send one request; any transport failure becomes an `OpenRouterError` without the key."""
    headers = auth_headers(api_key) if api_key else {}
    failure: str | None = None
    try:
        return client.request(method, path, headers=headers, json=json_body)
    except httpx.HTTPError as exc:
        failure = f"{method} {path} failed: {type(exc).__name__}: {scrub(str(exc), api_key)}"
    except (UnicodeError, ValueError) as exc:
        # Raised while encoding the request (for example a non-ASCII header).
        # The message can quote the header value, so it is left out entirely.
        failure = f"{method} {path} failed: {type(exc).__name__} while encoding the request"
    # Raised outside the except block so the httpx exception is not chained.
    raise OpenRouterError(failure)


def read_json(response: httpx.Response, *, api_key: str | None = None) -> dict[str, Any]:
    """Return the JSON body of a successful response, or raise a scrubbed error."""
    path = response.request.url.path
    if response.is_error:
        # Scrub first, then cut: a cut key would no longer match the scrubber.
        detail = scrub(response_message(response), api_key)[:300] or response.reason_phrase
        raise OpenRouterError(
            f"{path} returned HTTP {response.status_code}: {detail}", response.status_code
        )
    try:
        data = response.json()
    except ValueError:
        data = None
    if not isinstance(data, dict):
        raise OpenRouterError(f"{path} returned a body that is not JSON", response.status_code)
    return data


def get_json(client: httpx.Client, path: str, *, api_key: str | None = None) -> dict[str, Any]:
    return read_json(send(client, "GET", path, api_key=api_key), api_key=api_key)


def response_message(response: httpx.Response) -> str:
    """The API's error message in a response, or its raw text. Not scrubbed:
    callers scrub it before they cut or show it."""
    try:
        error = response.json().get("error", {})
        message = error.get("message") if isinstance(error, dict) else error
    except (ValueError, AttributeError):
        message = None
    return str(message) if message else response.text
