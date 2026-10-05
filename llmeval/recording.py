"""`llmeval record`: record every planned call with the real API, safely and resumably.

The order of work:

1. The key: OPENROUTER_API_KEY must be set; nothing is sent without it. It
   comes from the environment only and is never printed.
2. The budget guard: the calls still to record are estimated
   (`llmeval.callplan`), and the run stops before any call when the estimate
   is above MAX_RUN_COST_USD. `:free` model ids cost $0.00.
3. The free quota: when the plan calls a `:free` model, `GET /api/v1/key`
   says how many free requests the key has left today. At 0 the run stops at
   once; otherwise it stops after that many (asking the endpoint once more
   first). When the number is unknown, HTTP 429 is the stop.
4. Pass 1: the system calls (rag and triage), in plan order, skipping every
   key already in the cassettes.
5. Pass 2: the judge calls, planned again from the recorded answers. A judge
   key exists only once the answers it grades are recorded; identical
   answers share a grading and need no pairwise question.
6. The manifest: written only when every planned call of both passes is in
   the cassettes. Until then the evaluation stays "pending first recorded
   run".

Every call is appended and flushed as soon as it returns, so a stopped run
loses nothing and a rerun continues where it stopped. The client keeps to the
configured requests per minute (`rpm`). A 429 with a short reset waits and
retries (at most three times); a 429 with a later reset, or without a reset
time, stops the run at once: a wait is never guessed. Every stop prints
`recorded X/Y` and how to continue. Exit codes of the command: 0 complete,
75 stopped on the free quota or a rate limit (rerun later), 1 an error.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
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
    quota_lines,
    to_record,
    up_to,
)
from llmeval.cassettes import CassetteStore, RunManifest, utc_now, write_manifest
from llmeval.client import ModelClient
from llmeval.config import Config, Mode, RoleConfig
from llmeval.datasets import file_sha256
from llmeval.openrouter import OpenRouterError, require_key
from llmeval.pricing import check_budget
from llmeval.quota import FreeQuotaUsed, QuotaExhausted, free_daily_quota
from llmeval.runner import PlannedRequest

# sysexits EX_TEMPFAIL: a temporary failure; rerunning later continues.
EXIT_STOPPED = 75
CONTINUE_HINT = "rerun make record to continue: the {n} recorded calls are kept and skipped"
UPPER_BOUND_NOTE = "the total counts judge calls at their upper bound until the answers exist"


class PlanMismatch(RuntimeError):
    """A request the client sent does not have the key the plan gave it."""


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
    ) -> None:
        self.client = client
        self.config = config
        self.echo = echo
        self.free_left = free_left
        self.refresh = refresh
        self.recorded = 0
        self.planned = 0
        self.exact = False
        self.sent = 0

    def progress(self, recorded: int, planned: int, exact: bool) -> None:
        self.recorded, self.planned, self.exact = recorded, planned, exact

    def _role(self, planned: PlannedRequest) -> RoleConfig:
        models = self.config.models
        return models.system if planned.role == "system" else models.judge

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
            role = self._role(request)
            if is_free(role.model):
                self._spend_free_request()
            result = self.client.complete(
                list(request.messages),
                role=role,
                response_format=request.response_format,
                repeat=request.repeat,
                tag=request.tag,
            )
            if result.key != request.key:
                raise PlanMismatch(
                    f"{request.tag.label(request.repeat)}: the request sent does not match "
                    "the plan; stopped (this is a bug in the planner)"
                )
            self.sent += 1
            self.recorded += 1
            label = request.tag.label(request.repeat)
            self.echo(f"recorded {self.recorded}/{self.planned}  {label}")


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

    Raises `MissingAPIKey` without a key and `BudgetExceeded` above the spend
    limit, both before any model call.
    """
    require_key(config.settings.api_key)
    models = config.models
    store = CassetteStore(cassettes_dir)
    plan = full_plan(inputs, store)
    counts = count_plan(plan, store, models)
    estimate = estimate_remaining_cost(plan, store, models, transport=transport, now=now)
    echo(estimate.headline())
    check_budget(estimate.usd, config.settings.max_run_cost_usd)
    for line in quota_lines(counts, models):
        echo(line)
    echo(
        f"to record: {up_to(counts.to_record, not counts.exact)} of "
        f"{up_to(counts.total, not counts.exact)} calls, at most {models.rpm} per minute; "
        "a 429 without a reset time stops the run at once"
    )
    free_left = None
    if counts.free_to_record:
        free_left = _free_requests_left(config, transport, echo)
        if free_left is not None and free_left < counts.free_to_record:
            echo(
                f"the key has {free_left} free-model requests left today; "
                "the recording stops after them"
            )

    def refresh() -> int | None:
        return _free_requests_left(config, transport, echo)

    with ModelClient(
        Mode.RECORD, store, config, transport, limiter=limiter, now=now, sleep=sleep
    ) as client:
        session = _Session(client, config, echo, free_left, refresh)
        session.progress(counts.recorded, counts.total, counts.exact)
        try:
            system = [p for p in to_record(plan, store) if p.role == "system"]
            if system:
                echo(
                    f"pass 1: {len(system)} system calls to record; until their answers exist, "
                    "the total counts judge calls at their upper bound"
                )
            session.send_all(system)
            plan = full_plan(inputs, store)
            counts = count_plan(plan, store, models)
            session.progress(counts.recorded, counts.total, counts.exact)
            judge = to_record(plan, store)
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
            if type(exc) is QuotaExhausted and exc.reset_at is None:
                echo(
                    "HTTP 429 came without a reset time, so the run stopped at once "
                    "instead of guessing a wait"
                )
            echo(CONTINUE_HINT.format(n=session.recorded))
            return RecordOutcome(
                False, session.recorded, session.planned, session.sent, stopped=stopped
            )
        except OpenRouterError as exc:
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
    if not session.sent:
        echo("nothing left to record")
    if counts.to_record or not counts.exact:
        echo(f"recorded {counts.recorded}/{counts.total}: not complete, no manifest written")
        return RecordOutcome(False, counts.recorded, counts.total, session.sent)
    path = write_manifest(cassettes_dir, build_manifest(inputs, plan, store, dataset_paths))
    echo(f"every planned call is recorded ({counts.total}); wrote {path}")
    return RecordOutcome(True, counts.recorded, counts.total, session.sent, manifest=path)
