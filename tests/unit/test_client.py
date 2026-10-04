"""The model client in its three modes: live, record and replay.

Synthetic data: every chat completion below is a made-up response served by an
httpx MockTransport, and the key is a made-up string. These tests write
cassettes only into pytest's temporary directories, never into `cassettes/`.
"""

import inspect
import json
import socket
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from pydantic import SecretStr

from llmeval.cassettes import CallTag, CassetteStore, request_key
from llmeval.client import CallResult, MissingRecording, ModelClient, build_request
from llmeval.config import Config, Mode, ModelsConfig, RoleConfig, Settings
from llmeval.openrouter import MissingAPIKey, OpenRouterError
from llmeval.quota import QuotaExhausted, RateLimiter

FAKE_KEY = "synthetic-test-key-0123456789abcdef"
MODEL = "vendor-a/small:free"
NOW = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
MESSAGES = [
    {"role": "system", "content": "You are a test double."},
    {"role": "user", "content": "Synthetic question?"},
]
TAG = CallTag(function="rag", case="rag-001", version="v1")


def make_config(api_key: str | None = FAKE_KEY) -> Config:
    models = ModelsConfig(
        system=RoleConfig(model=MODEL, temperature=0.2, seed=7, max_tokens=100),
        judge=RoleConfig(model="vendor-b/large:free", temperature=0, seed=7, max_tokens=100),
        repeats=3,
        rpm=18,
    )
    key = SecretStr(api_key) if api_key else None
    return Config(models=models, settings=Settings(api_key=key))


def completion(content="Synthetic answer.", cost: float | None = 0.0) -> dict:
    usage = {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17}
    if cost is not None:
        usage["cost"] = cost
    return {
        "id": "gen-synthetic-1",
        "model": MODEL,
        "provider": "SyntheticProvider",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"content": content}}],
        "usage": usage,
    }


class Recorder:
    """A MockTransport handler that replays a scripted list of responses."""

    def __init__(self, *responses: httpx.Response) -> None:
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.responses.pop(0)

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self)


class NoNetwork:
    """A transport handler that fails the test if anything tries to use it."""

    def __init__(self) -> None:
        self.touched = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.touched += 1
        raise AssertionError(f"network used in replay: {request.url}")


class FakeTime:
    def __init__(self) -> None:
        self.monotonic = 100.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.monotonic

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.monotonic += seconds


def make_client(mode, store, transport=None, config=None, **overrides) -> ModelClient:
    fake = overrides.pop("fake_time", FakeTime())
    kwargs = {
        "limiter": RateLimiter(18, clock=fake.clock, sleep=fake.sleep),
        "now": lambda: NOW,
        "sleep": fake.sleep,
        **overrides,
    }
    return ModelClient(mode, store, config or make_config(), transport, **kwargs)


def ask(client: ModelClient, repeat=0, tag=TAG, **params) -> CallResult:
    values = {"model": MODEL, "temperature": 0.2, "seed": 7, "max_tokens": 100, **params}
    return client.complete(MESSAGES, repeat=repeat, tag=tag, **values)


def cassette_text(directory) -> str:
    return "".join(p.read_text(encoding="utf-8") for p in sorted(directory.glob("*.jsonl")))


def test_record_then_replay_round_trip(tmp_path):
    recorder = Recorder(httpx.Response(200, json=completion()))
    with make_client(Mode.RECORD, CassetteStore(tmp_path), recorder.transport) as client:
        recorded = ask(client)

    offline = NoNetwork()
    replay = make_client(Mode.REPLAY, CassetteStore(tmp_path), httpx.MockTransport(offline))
    replayed = ask(replay)

    assert replayed == recorded
    assert replayed.content == "Synthetic answer."
    assert replayed.usage.prompt_tokens == 12
    assert replayed.model_used == MODEL
    assert replayed.cost_source == "provider"
    assert offline.touched == 0


def test_record_appends_one_line_without_headers_or_key(tmp_path):
    recorder = Recorder(httpx.Response(200, json=completion()))
    with make_client(Mode.RECORD, CassetteStore(tmp_path), recorder.transport) as client:
        ask(client)

    text = cassette_text(tmp_path)
    entry = json.loads(text)

    assert text.count("\n") == 1
    assert (tmp_path / "rag-v1.jsonl").exists()
    assert entry["request"]["messages"] == MESSAGES
    assert entry["response"]["content"] == "Synthetic answer."
    assert "headers" not in entry
    assert FAKE_KEY not in text
    assert "authorization" not in text.lower()


def test_record_sends_the_key_only_as_a_header(tmp_path):
    recorder = Recorder(httpx.Response(200, json=completion()))
    with make_client(Mode.RECORD, CassetteStore(tmp_path), recorder.transport) as client:
        ask(client)

    request = recorder.requests[0]
    assert request.method == "POST"
    assert request.url.path == "/api/v1/chat/completions"
    assert request.headers["authorization"] == f"Bearer {FAKE_KEY}"
    assert FAKE_KEY not in request.content.decode()


def test_record_skips_calls_already_recorded(tmp_path):
    recorder = Recorder(httpx.Response(200, json=completion()))
    with make_client(Mode.RECORD, CassetteStore(tmp_path), recorder.transport) as client:
        first = ask(client)
        again = ask(client)

    assert again == first
    assert len(recorder.requests) == 1


def test_each_repeat_is_its_own_recording(tmp_path):
    recorder = Recorder(*[httpx.Response(200, json=completion(f"Answer {n}.")) for n in range(3)])
    with make_client(Mode.RECORD, CassetteStore(tmp_path), recorder.transport) as client:
        answers = [ask(client, repeat=n) for n in range(3)]

    assert [a.content for a in answers] == ["Answer 0.", "Answer 1.", "Answer 2."]
    assert len({a.key for a in answers}) == 3
    assert [a.repeat for a in answers] == [0, 1, 2]


def test_live_writes_nothing(tmp_path):
    recorder = Recorder(httpx.Response(200, json=completion()))
    with make_client(Mode.LIVE, CassetteStore(tmp_path), recorder.transport) as client:
        result = ask(client)

    assert result.content == "Synthetic answer."
    assert list(tmp_path.iterdir()) == []


def test_live_and_record_need_a_key(tmp_path):
    for mode in (Mode.LIVE, Mode.RECORD):
        with pytest.raises(MissingAPIKey, match="OPENROUTER_API_KEY"):
            make_client(mode, CassetteStore(tmp_path), config=make_config(api_key=None))


def test_replay_miss_says_what_is_missing_and_what_to_do(tmp_path):
    client = make_client(Mode.REPLAY, CassetteStore(tmp_path), config=make_config(api_key=None))

    with pytest.raises(MissingRecording) as caught:
        ask(client, repeat=2)

    assert str(caught.value) == "no recording for rag-001/v1/2: run make record"
    assert caught.value.tag == TAG
    assert caught.value.repeat == 2


def test_replay_miss_without_tag_still_says_what_to_do(tmp_path):
    client = make_client(Mode.REPLAY, CassetteStore(tmp_path), config=make_config(api_key=None))

    with pytest.raises(MissingRecording, match="run make record"):
        ask(client, tag=None)


def test_replay_never_opens_a_socket(tmp_path, monkeypatch):
    recorder = Recorder(httpx.Response(200, json=completion()))
    with make_client(Mode.RECORD, CassetteStore(tmp_path), recorder.transport) as client:
        expected = ask(client)

    def no_socket(*args, **kwargs):
        raise AssertionError("replay tried to open a socket")

    monkeypatch.setattr(socket, "socket", no_socket)
    monkeypatch.setattr(socket, "create_connection", no_socket)
    # No transport, no key: the client would use real HTTP if it used any.
    replay = make_client(Mode.REPLAY, CassetteStore(tmp_path), config=make_config(api_key=None))

    assert ask(replay) == expected
    with pytest.raises(MissingRecording):
        ask(replay, repeat=1)


def test_cost_from_provider(tmp_path):
    recorder = Recorder(httpx.Response(200, json=completion(cost=0.00042)))
    with make_client(Mode.RECORD, CassetteStore(tmp_path), recorder.transport) as client:
        result = ask(client)

    entry = json.loads(cassette_text(tmp_path))
    assert result.cost_usd == pytest.approx(0.00042)
    assert result.cost_source == "provider"
    assert entry["prices"] is None
    assert [r.url.path for r in recorder.requests] == ["/api/v1/chat/completions"]


def test_cost_from_published_prices_when_provider_sends_none(tmp_path):
    models = {"data": [{"id": MODEL, "pricing": {"prompt": "0.000001", "completion": "0.000002"}}]}
    recorder = Recorder(
        httpx.Response(200, json=completion(cost=None)), httpx.Response(200, json=models)
    )
    with make_client(Mode.RECORD, CassetteStore(tmp_path), recorder.transport) as client:
        result = ask(client)

    entry = json.loads(cassette_text(tmp_path))
    assert result.cost_source == "published_prices"
    assert result.cost_usd == pytest.approx(12 * 0.000001 + 5 * 0.000002)
    assert entry["prices"]["prompt"] == "0.000001"
    assert [r.url.path for r in recorder.requests][-1] == "/api/v1/models"


def test_structured_output_requires_supporting_endpoints(tmp_path):
    schema = {"type": "json_schema", "json_schema": {"name": "t", "strict": True, "schema": {}}}
    recorder = Recorder(httpx.Response(200, json=completion("{}")))
    with make_client(Mode.LIVE, CassetteStore(tmp_path), recorder.transport) as client:
        ask(client, response_format=schema)

    body = json.loads(recorder.requests[0].content)
    assert body["response_format"] == schema
    assert body["provider"] == {"require_parameters": True}


def test_request_body_omits_unset_options():
    body = build_request(MESSAGES, model=MODEL, temperature=0, seed=None, max_tokens=50)

    assert body == {"model": MODEL, "messages": MESSAGES, "temperature": 0, "max_tokens": 50}


BASE_REQUEST = {
    "messages": MESSAGES,
    "model": MODEL,
    "temperature": 0.2,
    "seed": 7,
    "max_tokens": 100,
    "response_format": None,
}
CHANGED_REQUEST = {
    "messages": [{"role": "user", "content": "Another synthetic question?"}],
    "model": "vendor-b/other:free",
    "temperature": 0.3,
    "seed": 8,
    "max_tokens": 101,
    "response_format": {"type": "json_object"},
}


def test_every_build_request_keyword_is_covered_below():
    keywords = set(inspect.signature(build_request).parameters)

    assert keywords == set(BASE_REQUEST) == set(CHANGED_REQUEST)


@pytest.mark.parametrize("keyword", sorted(CHANGED_REQUEST))
def test_every_build_request_keyword_changes_the_key(keyword):
    base = build_request(**BASE_REQUEST)
    changed = build_request(**{**BASE_REQUEST, keyword: CHANGED_REQUEST[keyword]})

    assert request_key(changed, repeat=0) != request_key(base, repeat=0)


def test_integer_and_float_temperature_give_one_key():
    as_int = build_request(**{**BASE_REQUEST, "temperature": 0})
    as_float = build_request(**{**BASE_REQUEST, "temperature": 0.0})

    assert request_key(as_int, repeat=0) == request_key(as_float, repeat=0)


def test_latency_is_measured(tmp_path):
    ticks = iter([10.0, 10.25])
    recorder = Recorder(httpx.Response(200, json=completion()))
    with make_client(
        Mode.LIVE, CassetteStore(tmp_path), recorder.transport, timer=lambda: next(ticks)
    ) as client:
        result = ask(client)

    assert result.latency_ms == pytest.approx(250.0)
    assert result.recorded_at == NOW


def test_short_429_is_waited_out_and_retried(tmp_path):
    reset = str(int((NOW + timedelta(seconds=20)).timestamp() * 1000))
    recorder = Recorder(
        httpx.Response(429, headers={"X-RateLimit-Reset": reset}, json={"error": "slow down"}),
        httpx.Response(200, json=completion()),
    )
    fake = FakeTime()
    with make_client(
        Mode.RECORD, CassetteStore(tmp_path), recorder.transport, fake_time=fake
    ) as client:
        result = ask(client)

    assert result.content == "Synthetic answer."
    assert 20 in fake.sleeps
    assert len(CassetteStore(tmp_path)) == 1


def test_daily_429_stops_and_writes_nothing(tmp_path):
    reset = str(int((NOW + timedelta(hours=10)).timestamp() * 1000))
    recorder = Recorder(httpx.Response(429, headers={"X-RateLimit-Reset": reset}, json={}))
    with (
        make_client(Mode.RECORD, CassetteStore(tmp_path), recorder.transport) as client,
        pytest.raises(QuotaExhausted, match="rerun after 2026-01-01 22:00 UTC"),
    ):
        ask(client)

    assert list(tmp_path.iterdir()) == []


def test_http_error_does_not_leak_the_key_or_write(tmp_path):
    def echo(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": {"message": request.headers["authorization"]}})

    with (
        make_client(Mode.RECORD, CassetteStore(tmp_path), httpx.MockTransport(echo)) as client,
        pytest.raises(OpenRouterError) as caught,
    ):
        ask(client)

    assert FAKE_KEY not in str(caught.value)
    assert caught.value.__context__ is None
    assert list(tmp_path.iterdir()) == []


def test_response_without_choices_is_an_error(tmp_path):
    recorder = Recorder(httpx.Response(200, json={"error": {"message": "upstream failed"}}))
    with (
        make_client(Mode.LIVE, CassetteStore(tmp_path), recorder.transport) as client,
        pytest.raises(OpenRouterError, match="upstream failed"),
    ):
        ask(client)
