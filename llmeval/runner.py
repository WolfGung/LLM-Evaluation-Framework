"""The runner: plan the calls, run every case through the system, apply the checks.

One run covers datasets x prompt versions x repeats. Every model call goes
through the client (`ModelClient` or any `ChatModel`), so a replay run needs
no key and no network. The results of one function and prompt version go to
`results/<function>-<version>.json`: a record per case and repeat with every
check, and pass rates per layer.

When results are written:

- replay with `cassettes/manifest.json` (a complete recording): yes, and a
  missing cassette entry raises `MissingRecording` instead of being skipped;
- replay without a manifest: no. The run is "pending first recorded run",
  calls nothing and writes nothing;
- record or live: yes, and the file says which mode produced it.

The results file is a pure function of the replies, so a replay run
reproduces it byte for byte.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app import assistant
from app import triage as triage_app
from app.prompting import ChatModel, load_prompt, prompt_versions
from app.retrieval import BM25Index, default_index
from llmeval.cassettes import PENDING_RECORDED_RUN, CallTag, load_manifest, request_key
from llmeval.checks import CheckResult
from llmeval.checks import deterministic as det
from llmeval.checks import reference as ref
from llmeval.checks.retrieval import retrieval_recall
from llmeval.client import build_role_request
from llmeval.config import Mode, RoleConfig
from llmeval.datasets import RagCase, TriageCase, file_sha256
from llmeval.results import (
    CallRecord,
    CaseRecord,
    CheckRecord,
    FunctionResults,
    RunRecord,
    summarise,
    write_results,
)

EVAL_FUNCTIONS = ("rag", "triage")
# The prompt files behind each evaluated function (app/prompts/<name>_<v>.md).
PROMPT_NAMES = {"rag": "assistant", "triage": "triage"}
CASSETTES_DIR = Path("cassettes")


def versions_of(function: str) -> tuple[str, ...]:
    return prompt_versions(PROMPT_NAMES[function])


# --- plan -------------------------------------------------------------------


@dataclass(frozen=True)
class PlannedRequest:
    """One model call the run will make, with its cassette key."""

    function: str
    case_id: str
    version: str
    repeat: int
    messages: tuple[dict[str, str], ...]
    response_format: dict[str, Any] | None
    key: str

    @property
    def tag(self) -> CallTag:
        return CallTag(function=self.function, case=self.case_id, version=self.version)


def _versions(versions: Mapping[str, Sequence[str]] | None, function: str) -> tuple[str, ...]:
    known = versions_of(function)
    chosen = tuple(versions[function]) if versions and function in versions else known
    unknown = [v for v in chosen if v not in known]
    if unknown:
        raise ValueError(
            f"unknown prompt version for {function}: {', '.join(unknown)} "
            f"(known: {', '.join(known)})"
        )
    return chosen


def plan_requests(
    role: RoleConfig,
    *,
    rag_cases: Sequence[RagCase] = (),
    triage_cases: Sequence[TriageCase] = (),
    versions: Mapping[str, Sequence[str]] | None = None,
    repeats: int = 1,
    index: BM25Index | None = None,
) -> list[PlannedRequest]:
    """Every call of a run, in run order: function, version, case, repeat.

    Builds the exact messages the run sends, without calling anything, so the
    keys can be counted against the cassettes and the cost estimated.
    `versions` maps a function to the prompt versions to run (default: all).
    """
    plan: list[PlannedRequest] = []
    if rag_cases:
        for version in _versions(versions, "rag"):
            for case in rag_cases:
                messages, _ = assistant.prepare(case.question, version, index=index)
                plan += _planned("rag", case.id, version, messages, None, role, repeats)
    if triage_cases:
        for version in _versions(versions, "triage"):
            for case in triage_cases:
                messages, response_format = triage_app.prepare(case.text, version, role)
                plan += _planned(
                    "triage", case.id, version, messages, response_format, role, repeats
                )
    return plan


def _planned(
    function: str,
    case_id: str,
    version: str,
    messages: list[dict[str, str]],
    response_format: dict[str, Any] | None,
    role: RoleConfig,
    repeats: int,
) -> list[PlannedRequest]:
    body = build_role_request(messages, role, response_format)
    return [
        PlannedRequest(
            function=function,
            case_id=case_id,
            version=version,
            repeat=repeat,
            messages=tuple(messages),
            response_format=response_format,
            key=request_key(body, repeat),
        )
        for repeat in range(repeats)
    ]


# --- checks per run ----------------------------------------------------------


def _layer(layer: str, results: Iterable[CheckResult]) -> list[CheckRecord]:
    return [CheckRecord(layer=layer, **result.to_dict()) for result in results]


def rag_checks(case: RagCase, answer: assistant.AssistantAnswer) -> list[CheckRecord]:
    """Checks for one RAG answer, by layer. Which apply depends on `case.expected`."""
    text, cited, retrieved = answer.text, answer.cited_ids, answer.retrieved_ids
    retrieval = [retrieval_recall(case.expected_docs, retrieved)] if case.expected_docs else []
    deterministic = [det.has_text(text, answer.call.empty_reason)]
    if case.expected == "answer":
        deterministic.append(det.cites_retrieved(cited, retrieved))
    deterministic += [
        det.no_unretrieved_citations(cited, retrieved),
        det.no_forbidden(text, case.forbidden),
        det.within_length(text),
    ]
    if case.expected == "dont_know":
        sources = [hit.text for hit in answer.hits] + [case.question]
        deterministic.append(det.dont_know(text, sources))
    reference = [ref.required_facts(text, case.required_facts)] if case.expected == "answer" else []
    return (
        _layer("retrieval", retrieval)
        + _layer("deterministic", deterministic)
        + _layer("reference", reference)
    )


def triage_checks(case: TriageCase, raw: str) -> list[CheckRecord]:
    """Checks for one triage reply, by layer."""
    reference = [
        ref.category_match(str(case.category), raw),
        ref.priority_match(str(case.priority), raw),
        ref.order_id_match(case.order_id, raw),
    ]
    return _layer("deterministic", det.triage_checks(raw)) + _layer("reference", reference)


# --- running -----------------------------------------------------------------


def run_rag(
    client: ChatModel,
    role: RoleConfig,
    cases: Sequence[RagCase],
    version: str,
    *,
    repeats: int,
    index: BM25Index | None = None,
) -> list[CaseRecord]:
    records = []
    for case in cases:
        runs = []
        for repeat in range(repeats):
            answer = assistant.answer(
                client, role, case.question, version, index=index, repeat=repeat, case=case.id
            )
            runs.append(
                RunRecord(
                    repeat=repeat,
                    output=answer.text,
                    retrieved=list(answer.retrieved_ids),
                    cited=list(answer.cited_ids),
                    call=CallRecord.from_call(answer.call),
                    checks=rag_checks(case, answer),
                )
            )
        expected = case.model_dump(
            mode="json", include={"expected", "required_facts", "expected_docs", "forbidden"}
        )
        records.append(
            CaseRecord(
                id=case.id,
                category=case.category,
                input=case.question,
                expected=expected,
                runs=runs,
            )
        )
    return records


def run_triage(
    client: ChatModel,
    role: RoleConfig,
    cases: Sequence[TriageCase],
    version: str,
    *,
    repeats: int,
) -> list[CaseRecord]:
    records = []
    for case in cases:
        runs = []
        for repeat in range(repeats):
            error: str | None = None
            try:
                answer = triage_app.triage(
                    client, role, case.text, version, repeat=repeat, case=case.id
                )
            except triage_app.TriageError as exc:
                # An invalid reply is a result to check, not a crash.
                assert exc.call is not None  # triage() always attaches the call
                raw, call, error = exc.raw, exc.call, exc.kind
            else:
                raw, call = answer.call.content, answer.call
            runs.append(
                RunRecord(
                    repeat=repeat,
                    output=raw,
                    error=error,
                    call=CallRecord.from_call(call),
                    checks=triage_checks(case, raw),
                )
            )
        expected = {
            "category": str(case.category),
            "priority": str(case.priority),
            "order_id": case.order_id,
        }
        records.append(
            CaseRecord(
                id=case.id,
                category=str(case.category),
                input=case.text,
                expected=expected,
                runs=runs,
            )
        )
    return records


def evaluate(
    function: str,
    version: str,
    client: ChatModel,
    role: RoleConfig,
    cases: Sequence[RagCase] | Sequence[TriageCase],
    *,
    mode: Mode,
    dataset_path: Path,
    repeats: int,
    index: BM25Index | None = None,
) -> FunctionResults:
    """Run one function with one prompt version over its dataset."""
    if function == "rag":
        records = run_rag(client, role, cases, version, repeats=repeats, index=index)
    elif function == "triage":
        records = run_triage(client, role, cases, version, repeats=repeats)
    else:
        raise ValueError(f"unknown function {function!r}; known: {', '.join(EVAL_FUNCTIONS)}")
    prompt = load_prompt(PROMPT_NAMES[function], version)
    return FunctionResults(
        function=function,
        version=version,
        mode=str(mode),
        model=role.model,
        prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        dataset=Path(dataset_path).name,
        dataset_sha256=file_sha256(dataset_path),
        repeats=repeats,
        summary=summarise(function, records),
        cases=records,
    )


@dataclass(frozen=True)
class RunOutcome:
    pending: bool
    reason: str | None
    results: tuple[FunctionResults, ...]
    written: tuple[Path, ...]


def run(
    client: ChatModel,
    role: RoleConfig,
    *,
    mode: Mode,
    cassettes_dir: Path | str,
    results_dir: Path | str,
    rag_cases: Sequence[RagCase] = (),
    triage_cases: Sequence[TriageCase] = (),
    dataset_paths: Mapping[str, Path],
    versions: Mapping[str, Sequence[str]] | None = None,
    repeats: int = 1,
    index: BM25Index | None = None,
) -> RunOutcome:
    """Evaluate every function that has cases, for each chosen prompt version.

    Replay without a manifest returns a pending outcome and calls nothing.
    Results are written only after every function and version has run, so a
    failure (such as a missing recording) leaves `results/` untouched.
    """
    mode = Mode(mode)
    if mode is Mode.REPLAY and load_manifest(cassettes_dir) is None:
        return RunOutcome(pending=True, reason=PENDING_RECORDED_RUN, results=(), written=())
    index = index or default_index()
    work: list[tuple[str, Sequence[RagCase] | Sequence[TriageCase]]] = []
    if rag_cases:
        work.append(("rag", rag_cases))
    if triage_cases:
        work.append(("triage", triage_cases))
    results = tuple(
        evaluate(
            function,
            version,
            client,
            role,
            cases,
            mode=mode,
            dataset_path=dataset_paths[function],
            repeats=repeats,
            index=index,
        )
        for function, cases in work
        for version in _versions(versions, function)
    )
    written = tuple(write_results(result, results_dir) for result in results)
    return RunOutcome(pending=False, reason=None, results=results, written=written)
