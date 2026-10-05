"""`llmeval record`: record every planned call with the real API, safely and resumably.

The order of work:

1. The key: OPENROUTER_API_KEY must be set; nothing is sent without it. It
   comes from the environment only and is never printed. Then the lock: one
   record run at a time per cassettes directory (`.record.lock`); a second
   run is refused instead of appending to the same files.
2. The budget guard: the calls still to record are estimated
   (`llmeval.callplan`), and the run stops before any call when the estimate
   is above MAX_RUN_COST_USD. `:free` model ids cost $0.00. The estimate can
   come from recorded costs, which are not a bound, so the run also keeps a
   running spending cap (`SpendCap`): before each paid call it adds the
   call's published-price bound (full token budget) to what this run has
   spent and stops if that would pass MAX_RUN_COST_USD; a call whose cost is
   unknown counts at that bound. A paid model with no published price is
   refused before any call. When real costs exceed the published bound, the
   run stops right after the call that passed the limit.
3. The free quota: when the plan calls a `:free` model, `GET /api/v1/key`
   says how many free requests the key has left today. At 0 the run stops at
   once; otherwise it stops after that many (asking the endpoint once more
   first). When the number is unknown, HTTP 429 is the stop.
4. Pass 1: the system calls (rag and triage), skipping every key already in
   the cassettes. One call of each kind goes first (rag v1, rag v2, triage
   v1, triage v2), then the rest in plan order, so a broken prompt or config
   shows on the first day of a recording spread over many. Calls an earlier
   run tried and could not record go after every fresh call of both passes
   (`split_tried`), followed by the judge calls their answers make ready.
5. Pass 2: the judge calls, planned again from the recorded answers, one
   grading and one pairwise question first. A judge key exists only once the
   answers it grades are recorded; identical answers share a grading and
   need no pairwise question.
6. The manifest: written only when every planned call of both passes is in
   the cassettes. Until then the evaluation stays "pending first recorded
   run".

A request the API refuses or fails (for example a moderation 403 on one
prompt) is skipped with its reason and the run goes on. Failures are counted
per distinct request body (`body_id`): the other repeats of a failed request,
which send the same body, are not sent in this run, so one refused prompt
costs one request, not `repeats`. Skipped and not-sent calls are listed at the
end of every run, also when a stop ends it, and the next run asks them again.
Three stops guard against a broken setup:

- `MAX_CONSECUTIVE_FAILURES` (3) different requests failing in a row whose
  kinds (function, version and role) are not proven. A kind is proven when
  one of its calls succeeded in this run, or when one of its planned keys is
  already recorded: a key hashes the whole request (model, parameters,
  prompt), so a recorded key of the current plan proves the current config
  for that kind, while a changed model id leaves its kind unproven. A wrong
  model id or config fails the first call of its kind, and the day-1
  ordering probes every kind first. A refusal on a proven kind (one refused
  prompt, a block of them, on any day) never counts toward this stop.
- `MAX_FAILURES_IN_A_ROW` (20) different requests failing in a row with no
  HTTP response or a 5xx: the API or the network may be down. A 4xx is an
  answer about that one request and does not count.
- HTTP 401 or 402 stops at once: the key is not accepted, or the account has
  no credit for the request.

Every call is appended and flushed as soon as it returns, so a stopped run
loses nothing and a rerun continues where it stopped. The client keeps to the
configured requests per minute (`rpm`). A 429 with a short reset waits and
retries (at most three times); a 429 with a later reset, or without a reset
time, stops the run at once: a wait is never guessed. Every stop prints
`recorded X/Y` and how to continue; after a 429 without a reset time the key
endpoint is read once more to say whether the daily quota is the cause. Exit
codes of the command: 0 complete, 75 stopped on the free quota or a rate
limit (rerun later), 1 an error.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from llmeval.callplan import (
    PlanInputs,
    count_plan,
    estimate_remaining_cost,
    full_plan,
    is_free,
    plural,
    quota_lines,
    to_record,
    up_to,
)
from llmeval.cassettes import CassetteStore, RunManifest, request_key, utc_now, write_manifest
from llmeval.client import ModelClient, build_role_request
from llmeval.config import Config, Mode, RoleConfig
from llmeval.datasets import file_sha256
from llmeval.openrouter import OpenRouterError, require_key
from llmeval.pricing import (
    ModelPrice,
    PlannedCall,
    PricingError,
    check_budget,
    estimate_cost,
    estimate_prompt_tokens,
    fetch_prices,
    format_usd,
)
from llmeval.quota import FreeQuotaUsed, QuotaExhausted, RateLimitedNoReset, free_daily_quota
from llmeval.runner import PlannedRequest

# sysexits EX_TEMPFAIL: a temporary failure; rerunning later continues.
EXIT_STOPPED = 75
CONTINUE_HINT = "rerun make record to continue: the {n} recorded calls are kept and skipped"
UPPER_BOUND_NOTE = "the total counts judge calls at their upper bound until the answers exist"


LOCK_FILE = ".record.lock"
# Different requests of unproven kinds failing in a row that stop the run: a
# wrong model id or config fails every call.
MAX_CONSECUTIVE_FAILURES = 3
# Different requests failing in a row with no response or a 5xx that stop the
# run: the API or the network is likely down.
MAX_FAILURES_IN_A_ROW = 20


class PlanMismatch(RuntimeError):
    """A request the client sent does not have the key the plan gave it."""


class RecordLocked(RuntimeError):
    """Another record run holds the lock on the same cassettes directory,
    or the platform has no file locks (`fcntl`)."""


class RecordStop(RuntimeError):
    """Failures that stop the run (the message says why)."""


class TooManyFailures(RecordStop):
    """Too many different requests failed in a row (see the module docstring)."""


class AccessDenied(RecordStop):
    """HTTP 401 or 402: the key is not accepted, or the account has no credit."""


class SpendCapReached(RuntimeError):
    """The run stopped to keep its spending under MAX_RUN_COST_USD."""


class SpendCap:
    """What this run has spent on paid calls, kept under `limit` (USD).

    `prices` are the published prices of the paid models; a free model costs
    nothing and is never checked. `unknown` counts calls whose cost the API
    did not report (counted at their published-price bound, never as 0).
    """

    def __init__(self, limit: float, prices: Mapping[str, ModelPrice]) -> None:
        self.limit = limit
        self.prices = dict(prices)
        self.spent = 0.0
        self.unknown = 0

    def bound(self, role: RoleConfig, request: PlannedRequest) -> float:
        """The most this call can cost: its prompt and full budgets at published prices."""
        if is_free(role.model):
            return 0.0
        call = PlannedCall.for_role(role, estimate_prompt_tokens(request.messages))
        return estimate_cost([call], self.prices)

    def check(self, bound: float) -> None:
        if bound and self.spent + bound > self.limit:
            raise SpendCapReached(
                f"spending cap: this run spent {format_usd(self.spent)} of "
                f"MAX_RUN_COST_USD={format_usd(self.limit)}; the next call could cost up to "
                f"{format_usd(bound)}, so the run stopped before it"
            )

    def add(self, cost: float | None, *, bound: float) -> None:
        if cost is None:
            self.unknown += 1
            cost = bound
        self.spent += cost
        if self.spent > self.limit:
            raise SpendCapReached(
                f"spending cap: this run spent {format_usd(self.spent)} of "
                f"MAX_RUN_COST_USD={format_usd(self.limit)}; the last call cost more than its "
                "published-price bound, so the run stopped"
            )


def _cap_prices(
    plan: Sequence[PlannedRequest],
    store: CassetteStore,
    config: Config,
    transport: httpx.BaseTransport | None,
    now: Callable[[], datetime],
) -> dict[str, ModelPrice]:
    """Published prices of the paid models that still have calls to record."""
    models = config.models
    paid = sorted(
        {
            models.role(p.role).model
            for p in to_record(plan, store)
            if not is_free(models.role(p.role).model)
        }
    )
    if not paid:
        return {}
    try:
        return fetch_prices(paid, transport=transport, now=now)
    except (PricingError, OpenRouterError) as exc:
        raise PricingError(f"{exc}; the running spending cap needs it") from None


Kind = tuple[str, str, str]


def body_id(request: PlannedRequest, role: RoleConfig) -> str:
    """What identifies a request's body across repeats: its key at repeat 0.

    The repeats of a system call send the same body; two gradings with
    `judge_repeats: all` grade different answers, so their bodies differ.
    """
    body = build_role_request(list(request.messages), role, request.response_format)
    return request_key(body, 0)


def kind_of(request: PlannedRequest) -> Kind:
    """A request's kind: its function, prompt version (or version pair) and role.

    Every call of one kind is built from the same prompt and model, so a
    broken model id or config fails the first call of a kind.
    """
    return (request.function, request.version, request.role)


def kinds_first(planned: Sequence[PlannedRequest]) -> list[PlannedRequest]:
    """The first call of each kind (`kind_of`), then the rest in their order."""
    first: dict[Kind, PlannedRequest] = {}
    for request in planned:
        first.setdefault(kind_of(request), request)
    leaders = list(first.values())
    chosen = {id(request) for request in leaders}
    return leaders + [request for request in planned if id(request) not in chosen]


def split_tried(
    left: Sequence[PlannedRequest], plan: Sequence[PlannedRequest], store: CassetteStore
) -> tuple[list[PlannedRequest], list[PlannedRequest]]:
    """Split the calls to record into fresh ones and those an earlier run tried.

    No state is stored: an unrecorded call that comes before a recorded call
    of its kind in plan order was reached before and could not be recorded
    (earlier runs go in plan order). `record` sends those after every fresh
    call, so a block of refused prompts does not use the day's quota first.
    """
    position = {id(request): index for index, request in enumerate(plan)}
    last_recorded: dict[Kind, int] = {}
    for index, request in enumerate(plan):
        if request.key is not None and request.key in store:
            last_recorded[kind_of(request)] = index
    tried = [
        request
        for request in left
        if position[id(request)] < last_recorded.get(kind_of(request), -1)
    ]
    tried_ids = {id(request) for request in tried}
    return [request for request in left if id(request) not in tried_ids], tried


@contextmanager
def record_lock(cassettes_dir: Path) -> Iterator[None]:
    """Hold `<cassettes_dir>/.record.lock` for the whole run, or refuse at once.

    Two runs appending to the same cassette files would interleave lines and
    record calls twice. The lock is advisory (`flock`), released when the run
    ends or the process dies; the file itself stays and is git-ignored.
    """
    try:
        import fcntl  # POSIX only; imported here so that other commands work anywhere
    except ImportError:
        raise RecordLocked(
            "recording needs a POSIX system (Linux, macOS, or the Docker image) to lock the "
            "cassettes directory; the other commands work here"
        ) from None
    path = Path(cassettes_dir) / LOCK_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RecordLocked(
                f"another record run is active on {cassettes_dir} ({LOCK_FILE} is held); "
                "wait for it to finish"
            ) from None
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@dataclass(frozen=True)
class RecordOutcome:
    """How a record run ended.

    `recorded` of `planned` distinct calls are in the cassettes (`planned` is
    an upper bound while judge calls wait for their answers); `sent` calls
    were made by this run. `stopped` is set when the free quota or a rate
    limit stopped the run, `error` when the API failed.
    """

    complete: bool
    recorded: int
    planned: int
    sent: int
    manifest: Path | None = None
    stopped: QuotaExhausted | None = None
    error: str | None = None


class _Session:
    """Sends planned calls one at a time and keeps the progress count."""

    def __init__(
        self,
        client: ModelClient,
        config: Config,
        echo: Callable[[str], None],
        free_left: int | None,
        refresh: Callable[[], int | None],
        cap: SpendCap,
    ) -> None:
        self.client = client
        self.config = config
        self.echo = echo
        self.free_left = free_left
        self.refresh = refresh
        self.cap = cap
        self.recorded = 0
        self.planned = 0
        self.exact = False
        self.sent = 0
        self.skipped: list[tuple[str, str]] = []
        self.not_sent: list[tuple[str, str]] = []
        # Keys of the calls skipped or not sent in this run, and of every call
        # this run sent, skipped or held back.
        self.missing: set[str] = set()
        self.attempted: set[str] = set()
        self.failed: dict[str, str] = {}
        self.proven_kinds: set[Kind] = set()
        # Different requests failing in a row: of unproven kinds, and with no
        # response or a 5xx (an outage).
        self.unproven_failures = 0
        self.outage_failures = 0

    def progress(self, recorded: int, planned: int, exact: bool) -> None:
        self.recorded, self.planned, self.exact = recorded, planned, exact

    def prove(self, plan: Sequence[PlannedRequest], store: CassetteStore) -> None:
        """Mark the kinds with a planned key already recorded as proven."""
        for request in plan:
            if request.key is not None and request.key in store:
                self.proven_kinds.add(kind_of(request))

    def _spend_free_request(self) -> None:
        if self.free_left is None:
            return
        if self.free_left <= 0:
            # Ask once more: the limit may have grown or the day may have turned.
            self.free_left = self.refresh()
            if self.free_left is not None and self.free_left <= 0:
                raise FreeQuotaUsed(self.free_left)
            if self.free_left is None:
                return
        self.free_left -= 1

    def _fail(
        self, request: PlannedRequest, body: str, label: str, error: OpenRouterError
    ) -> None:
        """Skip a failed request and stop the run when failures look like a broken setup."""
        reason = str(error)
        self.failed[body] = reason
        self.skipped.append((label, reason))
        if request.key is not None:
            self.missing.add(request.key)
        self.echo(f"skipped {label}: {reason}")
        if error.status == 401:
            raise AccessDenied(
                f"the API key was not accepted ({reason}): check OPENROUTER_API_KEY"
            )
        if error.status == 402:
            raise AccessDenied(
                f"the account has no credit for this request ({reason}): add credit, "
                "or use :free model ids"
            )
        if kind_of(request) not in self.proven_kinds:
            self.unproven_failures += 1
        if error.status is None or error.status >= 500:
            self.outage_failures += 1
        if self.unproven_failures >= MAX_CONSECUTIVE_FAILURES:
            raise TooManyFailures(
                f"{self.unproven_failures} different requests in a row failed, and none of "
                f"their kinds has succeeded in this run: check the model id and config "
                f"(last: {reason})"
            )
        if self.outage_failures >= MAX_FAILURES_IN_A_ROW:
            raise TooManyFailures(
                f"{self.outage_failures} different requests in a row failed: the API or "
                f"network may be down; rerun later (last: {reason})"
            )

    def send_all(self, planned: Sequence[PlannedRequest]) -> None:
        for request in planned:
            if request.key is not None:
                self.attempted.add(request.key)
            label = request.tag.label(request.repeat)
            role = self.config.models.role(request.role)
            body = body_id(request, role)
            if body in self.failed:
                # Another repeat of this request failed in this run: do not ask again.
                reason = f"an earlier repeat failed ({self.failed[body]})"
                self.not_sent.append((label, reason))
                if request.key is not None:
                    self.missing.add(request.key)
                self.echo(f"not sent {label}: {reason}")
                continue
            bound = self.cap.bound(role, request)
            self.cap.check(bound)
            if is_free(role.model):
                self._spend_free_request()
            try:
                result = self.client.complete(
                    list(request.messages),
                    role=role,
                    response_format=request.response_format,
                    repeat=request.repeat,
                    tag=request.tag,
                )
            except OpenRouterError as exc:
                # The message is already scrubbed of the key (llmeval.openrouter).
                self._fail(request, body, label, exc)
                continue
            self.proven_kinds.add(kind_of(request))
            self.unproven_failures = self.outage_failures = 0
            if result.key != request.key:
                raise PlanMismatch(
                    f"{request.tag.label(request.repeat)}: the request sent does not match "
                    "the plan; stopped (this is a bug in the planner)"
                )
            self.sent += 1
            self.recorded += 1
            self.echo(f"recorded {self.recorded}/{self.planned}  {label}")
            self.cap.add(result.cost_usd, bound=bound)


def _summarise_skips(session: _Session, echo: Callable[[str], None]) -> None:
    """List the calls this run skipped and those it did not send."""
    if session.skipped:
        them = "them" if len(session.skipped) > 1 else "it"
        echo(
            f"skipped {plural(len(session.skipped), 'call')} "
            f"(rerun make record to retry {them}):"
        )
        for label, reason in session.skipped:
            echo(f"  {label}: {reason}")
    if session.not_sent:
        them = "them" if len(session.not_sent) > 1 else "it"
        echo(
            f"not sent {plural(len(session.not_sent), 'call')} "
            f"(rerun make record to send {them}):"
        )
        for label, reason in session.not_sent:
            echo(f"  {label}: {reason}")


def _free_requests_left(
    config: Config, transport: httpx.BaseTransport | None, echo: Callable[[str], None]
) -> int | None:
    try:
        quota = free_daily_quota(config.settings.api_key, transport=transport)
    except OpenRouterError as exc:
        echo(f"free quota today: not read ({exc}); the recording relies on HTTP 429 to stop")
        return None
    if quota is None or quota.remaining is None:
        echo("free quota today: unknown, so the recording relies on HTTP 429 to stop")
        return None
    echo(
        f"free quota today (GET /api/v1/key, read live): used {quota.used}, "
        f"limit {quota.limit}, remaining {quota.remaining}"
    )
    return quota.remaining


def _daily_quota_hint(config: Config, transport: httpx.BaseTransport | None) -> str:
    """After a 429 without a reset time: read the key once more to say whether
    the daily free quota is the cause."""
    try:
        quota = free_daily_quota(config.settings.api_key, transport=transport)
    except OpenRouterError as exc:
        return (
            f"the key endpoint could not be read ({exc}), so it is unknown whether the daily "
            "quota is used up: rerun later"
        )
    if quota is None or quota.remaining is None:
        return (
            "the key endpoint does not report free requests, so it is unknown whether the "
            "daily quota is used up: rerun later"
        )
    if quota.remaining > 0:
        return (
            f"the key still has {quota.remaining} free requests today, so this is not the "
            "daily quota: rerun in a few minutes"
        )
    return (
        "the key has no free requests left today, so this is the daily quota: "
        "rerun after the daily reset"
    )


def build_manifest(
    inputs: PlanInputs,
    plan: Sequence[PlannedRequest],
    store: CassetteStore,
    dataset_paths: Mapping[str, Path],
) -> RunManifest:
    """The manifest of a complete recording of `plan` (every key in `store`)."""
    keys = {p.key for p in plan}
    if None in keys:
        raise ValueError("a manifest needs every planned call to have a key")
    entries = [store.get(key) for key in keys if key is not None]
    if any(entry is None for entry in entries):
        raise ValueError("a manifest is written only when every planned call is recorded")
    recorded = [entry for entry in entries if entry is not None]
    functions = {
        "rag": inputs.rag_cases,
        "triage": inputs.triage_cases,
    }
    used = [function for function, cases in functions.items() if cases]
    models = inputs.models
    return RunManifest(
        models={"system": models.system.model, "judge": models.judge.model},
        prompt_versions={function: tuple(inputs.versions[function]) for function in used},
        repeats=models.repeats,
        datasets={
            dataset_paths[function].name: file_sha256(dataset_paths[function]) for function in used
        },
        recorded_from=min(entry.requested_at for entry in recorded),
        recorded_to=max(entry.recorded_at for entry in recorded),
        planned_calls=len(recorded),
        recorded_calls=len(recorded),
        stability_cases=models.stability_cases,
        judge_repeats=models.judge_repeats,
        rubric_sha256=inputs.rubric.sha256,
    )


def record_all(
    inputs: PlanInputs,
    config: Config,
    cassettes_dir: Path,
    dataset_paths: Mapping[str, Path],
    *,
    echo: Callable[[str], None] = print,
    transport: httpx.BaseTransport | None = None,
    limiter: Any = None,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], datetime] = utc_now,
) -> RecordOutcome:
    """Record every planned call (see the module docstring for the order of work).

    Raises `MissingAPIKey` without a key, `RecordLocked` while another run is
    active, and `BudgetExceeded` above the spend limit, all before any model
    call.
    """
    require_key(config.settings.api_key)
    with record_lock(cassettes_dir):
        return _record(
            inputs,
            config,
            cassettes_dir,
            dataset_paths,
            echo=echo,
            transport=transport,
            limiter=limiter,
            sleep=sleep,
            now=now,
        )


def _record(
    inputs: PlanInputs,
    config: Config,
    cassettes_dir: Path,
    dataset_paths: Mapping[str, Path],
    *,
    echo: Callable[[str], None],
    transport: httpx.BaseTransport | None,
    limiter: Any,
    sleep: Callable[[float], None],
    now: Callable[[], datetime],
) -> RecordOutcome:
    models = config.models
    store = CassetteStore(cassettes_dir)
    for notice in store.notices:
        echo(f"notice: {notice}")
    plan = full_plan(inputs, store)
    counts = count_plan(plan, store, models)
    estimate = estimate_remaining_cost(plan, store, models, transport=transport, now=now)
    for line in estimate.lines:
        echo(f"  {line}")
    echo(estimate.headline())
    limit = config.settings.max_run_cost_usd
    check_budget(estimate.usd, limit)
    echo(f"spend limit MAX_RUN_COST_USD: {format_usd(limit)}; the estimate is within it")
    for line in quota_lines(counts, models):
        echo(line)
    echo(
        f"to record: {up_to(counts.to_record, not counts.exact)} of "
        f"{up_to(counts.total, not counts.exact)} calls, at most {models.rpm} per minute; "
        "a 429 without a reset time stops the run at once"
    )
    free_left = None
    calls_free_models = counts.free_to_record > 0
    if calls_free_models:
        free_left = _free_requests_left(config, transport, echo)
        if free_left is not None and free_left < counts.free_to_record:
            echo(
                f"the key has {free_left} free-model requests left today; "
                "the recording stops after them"
            )

    def refresh() -> int | None:
        return _free_requests_left(config, transport, echo)

    cap = SpendCap(
        config.settings.max_run_cost_usd, _cap_prices(plan, store, config, transport, now)
    )
    store.cut_unfinished_lines()
    with ModelClient(
        Mode.RECORD, store, config, transport, limiter=limiter, now=now, sleep=sleep
    ) as client:
        session = _Session(client, config, echo, free_left, refresh, cap)
        session.progress(counts.recorded, counts.total, counts.exact)
        session.prove(plan, store)
        stopped: QuotaExhausted | None = None
        error: str | None = None
        hint = ""
        try:
            fresh, retry_system = split_tried(
                [p for p in to_record(plan, store) if p.role == "system"], plan, store
            )
            if fresh:
                echo(
                    f"pass 1: {len(fresh)} system calls to record; until their answers exist, "
                    "the total counts judge calls at their upper bound"
                )
            session.send_all(kinds_first(fresh))
            plan = full_plan(inputs, store)
            counts = count_plan(plan, store, models)
            session.progress(counts.recorded, counts.total, counts.exact)
            session.prove(plan, store)
            # Judge calls whose answers were skipped have no key and wait; skipped
            # system calls are not asked twice in one run.
            fresh, retry_judge = split_tried(
                [p for p in to_record(plan, store) if p.role == "judge" and p.key is not None],
                plan,
                store,
            )
            if fresh:
                echo(
                    f"pass 2: judge calls planned from the recorded answers: {len(fresh)} to "
                    f"record; total now {counts.total}"
                )
            session.send_all(kinds_first(fresh))
            # Calls an earlier run tried go last, after every fresh call.
            retries = retry_system + retry_judge
            if retries:
                echo(f"retrying {plural(len(retries), 'call')} an earlier run could not record")
                session.send_all(retries)
                plan = full_plan(inputs, store)
                counts = count_plan(plan, store, models)
                session.progress(counts.recorded, counts.total, counts.exact)
                ready = [
                    p
                    for p in to_record(plan, store)
                    if p.role == "judge" and p.key is not None and p.key not in session.attempted
                ]
                if ready:
                    echo(f"judge calls for the answers recorded on retry: {len(ready)} to record")
                    session.send_all(kinds_first(ready))
        except QuotaExhausted as exc:
            stopped = exc.with_progress(session.recorded, session.planned)
            echo(str(stopped))
            if not session.exact:
                echo(UPPER_BOUND_NOTE)
            if isinstance(exc, RateLimitedNoReset):
                echo(
                    "HTTP 429 came without a reset time, so the run stopped at once "
                    "instead of guessing a wait"
                )
                if calls_free_models:
                    echo(_daily_quota_hint(config, transport))
            hint = CONTINUE_HINT.format(n=session.recorded)
        except SpendCapReached as exc:
            echo(str(exc))
            error = str(exc)
            hint = (
                f"{session.recorded} of {session.planned} calls recorded; raise MAX_RUN_COST_USD "
                "or rerun make record later to continue: each run may spend up to the limit, "
                "and recorded calls are kept and skipped"
            )
        except RecordStop as exc:
            echo(f"stopped: {exc}")
            if not session.exact:
                echo(UPPER_BOUND_NOTE)
            error = str(exc)
            hint = (
                f"{session.recorded} of {session.planned} calls recorded; rerun make record "
                "to retry: recorded calls are kept and skipped"
            )
    _summarise_skips(session, echo)
    plan = full_plan(inputs, store)
    counts = count_plan(plan, store, models)
    # Only judge calls that grade an answer this run skipped or did not send;
    # the others wait for answers not reached yet (the upper-bound note).
    blocked = sum(p.key is None and bool(session.missing & set(p.grades)) for p in plan)
    if blocked:
        wait = "wait" if blocked > 1 else "waits"
        echo(f"{plural(blocked, 'judge call')} {wait} for answers that were skipped")
    if stopped is not None or error is not None:
        echo(hint)
        return RecordOutcome(
            False, session.recorded, session.planned, session.sent, stopped=stopped, error=error
        )
    if not session.sent and not session.skipped:
        echo("nothing left to record")
    if counts.to_record or not counts.exact:
        echo(f"recorded {counts.recorded}/{counts.total}: not complete, no manifest written")
        return RecordOutcome(False, counts.recorded, counts.total, session.sent)
    path = write_manifest(cassettes_dir, build_manifest(inputs, plan, store, dataset_paths))
    echo(f"every planned call is recorded ({counts.total}); wrote {path}")
    return RecordOutcome(True, counts.recorded, counts.total, session.sent, manifest=path)
