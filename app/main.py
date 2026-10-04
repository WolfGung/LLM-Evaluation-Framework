"""The Toolshop HTTP service.

    POST /assist  {question, version}  -> answer, citations, retrieved documents
    POST /triage  {text, version}      -> a validated TriageResult
    GET  /health

By default the model client comes from `config/models.yaml` and the
environment, in replay mode: answers come from `cassettes/`, no key is needed,
and a request without a recording gets 503 with the replay miss message.
`LLMEVAL_MODE=live` with `OPENROUTER_API_KEY` calls the model. The service
never records: cassettes come only from `make record`.

Tests pass their own model with `create_app(client=..., role=...)`.
"""

from __future__ import annotations

import threading
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, TypeVar

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import AfterValidator, BaseModel, ConfigDict, Field

from app import assistant, triage
from app.prompting import FUNCTIONS, ChatModel, prompt_versions
from app.retrieval import BM25Index, default_index
from app.triage import TriageError, TriageResult
from llmeval.cassettes import CassetteStore
from llmeval.client import MissingRecording, ModelClient
from llmeval.config import Config, Mode, RoleConfig, load_config
from llmeval.openrouter import OpenRouterError
from llmeval.quota import QuotaExhausted

CASSETTES_DIR = Path("cassettes")
SERVICE_MODES = (Mode.REPLAY, Mode.LIVE)
MAX_QUESTION_CHARS = 2000
MAX_TICKET_CHARS = 5000

T = TypeVar("T")


class ServiceConfigError(RuntimeError):
    """The service cannot start with this configuration."""


class _Request(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _not_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("must not be blank")
    return value


def _version_of(function: str) -> AfterValidator:
    def check(version: str) -> str:
        if version not in prompt_versions(function):
            known = ", ".join(prompt_versions(function))
            raise ValueError(f"unknown {function} prompt version {version!r}; known: {known}")
        return version

    return AfterValidator(check)


class AssistRequest(_Request):
    question: Annotated[
        str, Field(min_length=1, max_length=MAX_QUESTION_CHARS), AfterValidator(_not_blank)
    ]
    version: Annotated[str, _version_of("assistant")]


class AssistResponse(BaseModel):
    answer: str
    citations: list[str]
    retrieved: list[str]
    version: str
    model: str
    empty_reason: str | None


class TriageRequest(_Request):
    text: Annotated[
        str, Field(min_length=1, max_length=MAX_TICKET_CHARS), AfterValidator(_not_blank)
    ]
    version: Annotated[str, _version_of("triage")]


class TriageResponse(BaseModel):
    result: TriageResult
    version: str
    model: str


def build_client(config: Config, cassettes_dir: Path | str = CASSETTES_DIR) -> ModelClient:
    """The service's own model client: replay (default) or live, never record."""
    mode = config.settings.mode
    if mode not in SERVICE_MODES:
        raise ServiceConfigError(
            f"LLMEVAL_MODE={mode} is not available in the service (use replay or live); "
            "cassettes are recorded with make record"
        )
    return ModelClient(mode, CassetteStore(cassettes_dir), config)


def create_app(
    client: ChatModel | None = None,
    *,
    config: Config | None = None,
    role: RoleConfig | None = None,
    cassettes_dir: Path | str = CASSETTES_DIR,
    index: BM25Index | None = None,
) -> FastAPI:
    """Build the service.

    Nothing is loaded until the server starts, so importing this module needs
    no configuration. Without `client`, a `ModelClient` is built from `config`
    (or `config/models.yaml` and the environment); without `role`, the
    configured system role is used.
    """
    # One model call at a time: the client's rate limiter and HTTP connection
    # are shared, and a free model allows only a few requests per minute anyway.
    lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        loaded = config
        if loaded is None and (client is None or role is None):
            loaded = load_config()
        owned: ModelClient | None = None
        if client is None:
            owned = build_client(loaded, cassettes_dir)
        app.state.client = client if client is not None else owned
        app.state.role = role if role is not None else loaded.models.system
        app.state.mode = str(getattr(app.state.client, "mode", "custom"))
        app.state.index = index or default_index()
        try:
            yield
        finally:
            if owned is not None:
                owned.close()

    app = FastAPI(title="Toolshop", version="0.1.0", lifespan=lifespan)

    def with_model(request: Request, work: Callable[[ChatModel, RoleConfig], T]) -> T:
        with lock:
            return work(request.app.state.client, request.app.state.role)

    @app.get("/health")
    def health(request: Request) -> dict[str, Any]:
        return {
            "status": "ok",
            "mode": request.app.state.mode,
            "documents": len(request.app.state.index.ids),
            "prompt_versions": {
                function: list(prompt_versions(function)) for function in FUNCTIONS
            },
        }

    @app.post("/assist")
    def assist(body: AssistRequest, request: Request) -> AssistResponse:
        answer = with_model(
            request,
            lambda model, model_role: assistant.answer(
                model, model_role, body.question, body.version, index=request.app.state.index
            ),
        )
        return AssistResponse(
            answer=answer.text,
            citations=list(answer.cited_ids),
            retrieved=list(answer.retrieved_ids),
            version=body.version,
            model=answer.call.model_used,
            empty_reason=answer.call.empty_reason,
        )

    @app.post("/triage")
    def triage_ticket(body: TriageRequest, request: Request) -> TriageResponse:
        outcome = with_model(
            request,
            lambda model, model_role: triage.triage(model, model_role, body.text, body.version),
        )
        return TriageResponse(
            result=outcome.result, version=body.version, model=outcome.call.model_used
        )

    @app.exception_handler(MissingRecording)
    async def _missing(_: Request, exc: MissingRecording) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(QuotaExhausted)
    async def _quota(_: Request, exc: QuotaExhausted) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(OpenRouterError)
    async def _provider(_: Request, exc: OpenRouterError) -> JSONResponse:
        # OpenRouterError messages are scrubbed of the key where they are built.
        return JSONResponse(status_code=502, content={"detail": str(exc)})

    @app.exception_handler(TriageError)
    async def _triage(_: Request, exc: TriageError) -> JSONResponse:
        return JSONResponse(
            status_code=502, content={"detail": str(exc), "kind": exc.kind, "raw": exc.raw}
        )

    return app


app = create_app()
