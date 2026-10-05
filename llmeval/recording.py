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
   shows on the first day of a recording spread over many.
5. Pass 2: the judge calls, planned again from the recorded answers, one
   grading and one pairwise question first. A judge key exists only once the
   answers it grades are recorded; identical answers share a grading and
   need no pairwise question.
6. The manifest: written only when every planned call of both passes is in
   the cassettes. Until then the evaluation stays "pending first recorded
   run".

A request the API refuses or fails (for example a moderation 403 on one
prompt) is skipped with its reason and the run goes on; the skipped calls are
listed at the end and asked again by the next run. After
`MAX_CONSECUTIVE_FAILURES` (3) failures in a row the run stops, because a
wrong model id or config fails every call. Every call is appended and flushed
as soon as it returns, so a stopped run loses nothing and a rerun continues
where it stopped. The client keeps to the
configured requests per minute (`rpm`). A 429 with a short reset waits and
retries (at most three times); a 429 with a later reset, or without a reset
time, stops the run at once: a wait is never guessed. Every stop prints
`recorded X/Y` and how to continue; after a 429 without a reset time the key
endpoint is read once more to say whether the daily quota is the cause. Exit
codes of the command: 0 complete, 75 stopped on the free quota or a rate
limit (rerun later), 1 an error.
"""

from __future__ import annotations

import fcntl
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
from llmeval.cassettes import CassetteStore, RunManifest, utc_now, write_manifest
from llmeval.client import ModelClient
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
)
from llmeval.quota import FreeQuotaUsed, QuotaExhausted, RateLimitedNoReset, free_daily_quota
from llmeval.runner import PlannedRequest

# sysexits EX_TEMPFAIL: a temporary failure; rerunning later continues.
EXIT_STOPPED = 75
CONTINUE_HINT = "rerun make record to continue: the {n} recorded calls are kept and skipped"
UPPER_BOUND_NOTE = "the total counts judge calls at their upper bound until the answers exist"


LOCK_FILE = ".record.lock"
# Failed requests in a row that stop the run: one bad prompt is skipped, but a
# wrong model id or config fails every call and should stop early.
MAX_CONSECUTIVE_FAILURES = 3


class PlanMismatch(RuntimeError):
    """A request the client sent does not have the key the plan gave it."""


class RecordLocked(RuntimeError):
    """Another record run holds the lock on the same cassettes directory."""


class TooManyFailures(RuntimeError):
    """`MAX_CONSECUTIVE_FAILURES` requests failed in a row."""


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
                f"spending cap: this run spent ${self.spent:.4f} of "
                f"MAX_RUN_COST_USD=${self.limit:.4f}; the next call could cost up to "
                f"${bound:.4f}, so the run stopped before it"
            )

    def add(self, cost: float | None, *, bound: float) -> None:
        if cost is None:
            self.unknown += 1
            cost = bound
        self.spent += cost
        if self.spent > self.limit:
            raise SpendCapReached(
                f"spending cap: this run spent ${self.spent:.4f} of "
                f"MAX_RUN_COST_USD=${self.limit:.4f}; the last call cost more than its "
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


def kinds_first(planned: Sequence[PlannedRequest]) -> list[PlannedRequest]:
    """The first call of each kind, then the rest in their order.

    A system call's kind is its function and prompt version; a judge call's
    kind is grading or pairwise.
    """

    def kind(request: PlannedRequest) -> tuple[str, str]:
        return (request.function, request.version if request.role == "system" else "")

    first: dict[tuple[str, str], PlannedRequest] = {}
    for request in planned:
        first.setdefault(kind(request), request)
    leaders = list(first.values())
    chosen = {id(request) for request in leaders}
    return leaders + [request for request in planned if id(request) not in chosen]


@contextmanager
def record_lock(cassettes_dir: Path) -> Iterator[None]:
    """Hold `<cassettes_dir>/.record.lock` for the whole run, or refuse at once.

    Two runs appending to the same cassette files would interleave lines and
    record calls twice. The lock is advisory (`flock`), released when the run
    ends or the process dies; the file itself stays and is git-ignored.
    """
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
        self.failures_in_a_row = 0

    def progress(self, recorded: int, planned: int, exact: bool) -> None:
        self.recorded, self.planned, self.exact = recorded, planned, exact

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

    def send_all(self, planned: Sequence[PlannedRequest]) -> None:
        for request in planned:
            role = self.config.models.role(request.role)
            bound = self.cap.bound(role, request)
            self.cap.check(bound)
            if is_free(role.model):
                self._spend_free_request()
            label = request.tag.label(request.repeat)
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
                self.skipped.append((label, str(exc)))
                self.failures_in_a_row += 1
                self.echo(f"skipped {label}: {exc}")
                if self.failures_in_a_row >= MAX_CONSECUTIVE_FAILURES:
                    raise TooManyFailures(
                        f"{self.failures_in_a_row} calls in a row failed; a wrong model id or "
                        f"config fails every call (last: {exc})"
                    ) from None
                continue
            self.failures_in_a_row = 0
            if result.key != request.key:
                raise PlanMismatch(
                    f"{request.tag.label(request.repeat)}: the request sent does not match "
                    "the plan; stopped (this is a bug in the planner)"
                )
            self.sent += 1
            self.recorded += 1
            self.echo(f"recorded {self.recorded}/{self.planned}  {label}")
            self.cap.add(result.cost_usd, bound=bound)


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
    echo(f"spend limit MAX_RUN_COST_USD: ${limit:.2f}; the estimate is within it")
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
        try:
            system = kinds_first([p for p in to_record(plan, store) if p.role == "system"])
            if system:
                echo(
                    f"pass 1: {len(system)} system calls to record; until their answers exist, "
                    "the total counts judge calls at their upper bound"
                )
            session.send_all(system)
            plan = full_plan(inputs, store)
            counts = count_plan(plan, store, models)
            session.progress(counts.recorded, counts.total, counts.exact)
            # Judge calls whose answers were skipped have no key and wait; skipped
            # system calls are not asked twice in one run.
            judge = kinds_first(
                [p for p in to_record(plan, store) if p.role == "judge" and p.key is not None]
            )
            if judge:
                echo(
                    f"pass 2: judge calls planned from the recorded answers: {len(judge)} to "
                    f"record; total now {counts.total}"
                )
            session.send_all(judge)
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
            echo(CONTINUE_HINT.format(n=session.recorded))
            return RecordOutcome(
                False, session.recorded, session.planned, session.sent, stopped=stopped
            )
        except SpendCapReached as exc:
            echo(str(exc))
            echo(
                f"{session.recorded} of {session.planned} calls recorded; raise MAX_RUN_COST_USD "
                "or rerun make record later to continue: each run may spend up to the limit, "
                "and recorded calls are kept and skipped"
            )
            return RecordOutcome(
                False, session.recorded, session.planned, session.sent, error=str(exc)
            )
        except TooManyFailures as exc:
            echo(f"stopped: {exc}")
            echo(
                f"{session.recorded} of {session.planned} calls recorded; rerun make record "
                "to retry: recorded calls are kept and skipped"
            )
            if not session.exact:
                echo(UPPER_BOUND_NOTE)
            return RecordOutcome(
                False, session.recorded, session.planned, session.sent, error=str(exc)
            )
    plan = full_plan(inputs, store)
    counts = count_plan(plan, store, models)
    if not session.sent and not session.skipped:
        echo("nothing left to record")
    if session.skipped:
        them = "them" if len(session.skipped) > 1 else "it"
        echo(
            f"skipped {plural(len(session.skipped), 'call')} "
            f"(rerun make record to retry {them}):"
        )
        for label, reason in session.skipped:
            echo(f"  {label}: {reason}")
    if counts.waiting:
        wait = "wait" if counts.waiting > 1 else "waits"
        echo(f"{plural(counts.waiting, 'judge call')} {wait} for answers that were skipped")
    if counts.to_record or not counts.exact:
        echo(f"recorded {counts.recorded}/{counts.total}: not complete, no manifest written")
        return RecordOutcome(False, counts.recorded, counts.total, session.sent)
    path = write_manifest(cassettes_dir, build_manifest(inputs, plan, store, dataset_paths))
    echo(f"every planned call is recorded ({counts.total}); wrote {path}")
    return RecordOutcome(True, counts.recorded, counts.total, session.sent, manifest=path)
