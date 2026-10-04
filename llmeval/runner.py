"""The runner: plan the calls, run every case through the system, apply the checks.

One run covers datasets x prompt versions x repeats. Every model call goes
through the client (`ModelClient` or any `ChatModel`), so a replay run needs
no key and no network. The results of one function and prompt version are a
`FunctionResults` (see `llmeval.results`).

Every RAG answer gets the safety layer's leak checks (`llmeval.checks.safety`),
whatever its case: any question can retrieve a trap document. With a judge
role, the judge grades every run of every RAG case except the safety cases
(rules own safety), and compares the first RAG prompt version with each
later one, case by case, on repeat 0 (`PairwiseResults`). Judge calls go
through the same client, so they are recorded and replayed too.

Where results are written:

- replay with `cassettes/manifest.json` (a complete recording): to
  `results/`, and a missing cassette entry raises `MissingRecording` instead
  of being skipped;
- replay without a manifest: nowhere. The run is "pending first recorded
  run" and calls nothing;
- record: nowhere. Recording fills the cassettes; the results and the
  baseline come from a replay of the recorded run;
- live: to the git-ignored `results-live/`, never to `results/`.

The results file is a pure function of the replies, so a replay run
reproduces it byte for byte.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from app import assistant
from app import triage as triage_app
from app.prompting import ChatModel, load_prompt, prompt_versions
from app.retrieval import BM25Index, default_index
from llmeval.cassettes import (
    PENDING_RECORDED_RUN,
    CallTag,
    CassetteStore,
    load_manifest,
    repeats_for,
    request_key,
)
from llmeval.checks import CheckResult
from llmeval.checks import deterministic as det
from llmeval.checks import reference as ref
from llmeval.checks import safety as sf
from llmeval.checks.judge import (
    JUDGE_FUNCTION,
    PAIRWISE_FUNCTION,
    Judge,
    Rubric,
    compare_messages,
    compare_tag,
    grade_messages,
    grade_tag,
    load_rubric,
    pairwise_format,
    verdict_format,
)
from llmeval.checks.retrieval import retrieval_recall
from llmeval.client import build_role_request
from llmeval.config import Mode, RoleConfig
from llmeval.datasets import RagCase, TriageCase, file_sha256
from llmeval.results import (
    LIVE_RESULTS_DIR,
    CallRecord,
    CaseRecord,
    CheckRecord,
    FunctionResults,
    JudgeRecord,
    PairwiseCaseRecord,
    PairwiseResults,
    RunRecord,
    summarise,
    summarise_pairwise,
    write_live_results,
    write_results,
)

EVAL_FUNCTIONS = ("rag", "triage")
# The prompt files behind each evaluated function (app/prompts/<name>_<v>.md).
PROMPT_NAMES = {"rag": "assistant", "triage": "triage"}
CASSETTES_DIR = Path("cassettes")
# RAG categories the judge grades. Safety is never left to the judge: rules
# check it. Triage is not graded at all: its labels are checked by rules.
JUDGED_CATEGORIES = frozenset({"answerable", "multi_doc", "unanswerable"})


def versions_of(function: str) -> tuple[str, ...]:
    return prompt_versions(PROMPT_NAMES[function])


def judged(case: RagCase) -> bool:
    return case.category in JUDGED_CATEGORIES


# --- plan -------------------------------------------------------------------


@dataclass(frozen=True)
class PlannedRequest:
    """One model call the run will make, with its cassette key.

    `function` is rag, triage, judge (grading one answer) or pairwise (one of
    the two questions of a comparison). A judge or pairwise call grades
    answers that do not exist before they are recorded: `grades` holds the
    keys of the system calls it waits for, and `key` is None until every one
    of them is recorded. Until then `messages` hold the prompt with empty
    answers, enough to estimate its size.

    Two planned calls can share a key when their messages are equal, for
    example when two prompt versions wrote the same answer: the cassette
    holds one recording for both. Count recordings as distinct keys.
    """

    function: str
    case_id: str
    version: str
    repeat: int
    messages: tuple[dict[str, str], ...]
    response_format: dict[str, Any] | None
    key: str | None
    tag: CallTag
    grades: tuple[str, ...] = ()

    @property
    def role(self) -> Literal["system", "judge"]:
        """The model role from config/models.yaml that answers this call."""
        return "judge" if self.function in (JUDGE_FUNCTION, PAIRWISE_FUNCTION) else "system"


# The recorded answer of a system call, by its key, or None when it is not
# recorded yet (see `recorded_answers`).
Answers = Callable[[str], str | None]


def recorded_answers(store: CassetteStore) -> Answers:
    """Look up recorded answers in `store`, for planning the judge calls."""

    def answer(key: str) -> str | None:
        entry = store.get(key)
        return None if entry is None else entry.response.content

    return answer


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
    stability_cases: Collection[str] | None = None,
    index: BM25Index | None = None,
    judge: RoleConfig | None = None,
    rubric: Rubric | None = None,
    answers: Answers | None = None,
) -> list[PlannedRequest]:
    """Every call of a run: the system calls in run order (function, version,
    case, repeat), then the judge calls, then the pairwise calls.

    Builds the exact messages the run sends, without calling anything, so the
    keys can be counted against the cassettes and the cost estimated.
    `versions` maps a function to the prompt versions to run (default: all).
    `stability_cases` limits the repeats to those case ids (see `repeats_for`).
    With `judge`, the judge and pairwise calls are planned too; `answers`
    (see `recorded_answers`) supplies the recorded answers their keys need.
    """
    plan: list[PlannedRequest] = []
    rag_versions = _versions(versions, "rag") if rag_cases else ()
    for version in rag_versions:
        for case in rag_cases:
            messages, _ = assistant.prepare(case.question, version, index=index)
            n = repeats_for(case.id, repeats, stability_cases)
            plan += _planned("rag", case.id, version, messages, None, role, n)
    if triage_cases:
        for version in _versions(versions, "triage"):
            for case in triage_cases:
                messages, response_format = triage_app.prepare(case.text, version, role)
                n = repeats_for(case.id, repeats, stability_cases)
                plan += _planned("triage", case.id, version, messages, response_format, role, n)
    if judge is not None and rag_versions:
        rubric = rubric or load_rubric()
        graded = [case for case in rag_cases if judged(case)]
        rag_plan = [p for p in plan if p.function == "rag"]
        # System calls always have a key: their messages do not wait for anything.
        system = {(p.case_id, p.version, p.repeat): p.key for p in rag_plan if p.key}
        plan += _planned_gradings(rag_plan, graded, judge, rubric, answers, index)
        plan += _planned_comparisons(graded, rag_versions, system, judge, rubric, answers, index)
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
            tag=CallTag(function=function, case=case_id, version=version),
        )
        for repeat in range(repeats)
    ]


def _judge_key(
    messages: list[dict[str, str]],
    response_format: dict[str, Any],
    role: RoleConfig,
    repeat: int,
    ready: bool,
) -> str | None:
    return (
        request_key(build_role_request(messages, role, response_format), repeat) if ready else None
    )


def _planned_gradings(
    rag_plan: Sequence[PlannedRequest],
    cases: Sequence[RagCase],
    role: RoleConfig,
    rubric: Rubric,
    answers: Answers | None,
    index: BM25Index | None,
) -> list[PlannedRequest]:
    """One grading per planned answer of a judged case, with the same repeat."""
    questions = {case.id: case.question for case in cases}
    plan = []
    for system in rag_plan:
        if system.case_id not in questions:
            continue
        question = questions[system.case_id]
        _, hits = assistant.prepare(question, system.version, index=index)
        answer = answers(system.key) if answers and system.key else None
        messages = grade_messages(rubric, question, hits, answer or "")
        plan.append(
            PlannedRequest(
                function=JUDGE_FUNCTION,
                case_id=system.case_id,
                version=system.version,
                repeat=system.repeat,
                messages=tuple(messages),
                response_format=verdict_format(),
                key=_judge_key(messages, verdict_format(), role, system.repeat, answer is not None),
                tag=grade_tag(system.case_id, system.version),
                grades=(system.key,) if system.key else (),
            )
        )
    return plan


def _planned_comparisons(
    cases: Sequence[RagCase],
    versions: Sequence[str],
    system: Mapping[tuple[str, str, int], str],
    role: RoleConfig,
    rubric: Rubric,
    answers: Answers | None,
    index: BM25Index | None,
) -> list[PlannedRequest]:
    """Two questions per judged case and version pair, on the answers of repeat 0."""
    plan = []
    for pair in pairs_of(versions):
        for case in cases:
            _, hits = assistant.prepare(case.question, pair[0], index=index)
            keys = (system[(case.id, pair[0], 0)], system[(case.id, pair[1], 0)])
            texts = [answers(key) if answers else None for key in keys]
            ready = all(text is not None for text in texts)
            first, second = (text or "" for text in texts)
            for shown_as_a, a, b in ((pair[0], first, second), (pair[1], second, first)):
                messages = compare_messages(rubric, case.question, hits, a, b)
                plan.append(
                    PlannedRequest(
                        function=PAIRWISE_FUNCTION,
                        case_id=case.id,
                        version="-".join(pair),
                        repeat=0,
                        messages=tuple(messages),
                        response_format=pairwise_format(),
                        key=_judge_key(messages, pairwise_format(), role, 0, ready),
                        tag=compare_tag(case.id, pair, shown_as_a),
                        grades=keys,
                    )
                )
    return plan


def pairs_of(versions: Sequence[str]) -> list[tuple[str, str]]:
    """The version pairs compared: the first version with each later one."""
    return [(versions[0], later) for later in versions[1:]]


# --- checks per run ----------------------------------------------------------


def _layer(layer: str, results: Iterable[CheckResult]) -> list[CheckRecord]:
    return [CheckRecord(layer=layer, **result.to_dict()) for result in results]


def rag_checks(case: RagCase, answer: assistant.AssistantAnswer) -> list[CheckRecord]:
    """Checks for one RAG answer, by layer. Which apply depends on `case.expected`.

    The safety layer's leak checks run on every case, because any question
    can retrieve a trap document. A safety case checks its forbidden claims
    in the safety layer (`no_unsafe_claims`) instead of the deterministic
    one, and an off-topic case also checks that the answer refuses.
    """
    text, cited, retrieved = answer.text, answer.cited_ids, answer.retrieved_ids
    is_safety = case.category == "safety"
    retrieval = [retrieval_recall(case.expected_docs, retrieved)] if case.expected_docs else []
    deterministic = [det.has_text(text, answer.call.empty_reason)]
    if case.expected == "answer":
        deterministic.append(det.cites_retrieved(cited, retrieved))
    deterministic.append(det.no_unretrieved_citations(cited, retrieved))
    if not is_safety:
        deterministic.append(det.no_forbidden(text, case.forbidden))
    deterministic.append(det.within_length(text))
    if case.expected == "dont_know":
        sources = [hit.text for hit in answer.hits] + [case.question]
        deterministic.append(det.dont_know(text, sources))
    reference = [ref.required_facts(text, case.required_facts)] if case.expected == "answer" else []
    safety = sf.leak_checks(text, case.question)
    if is_safety:
        safety.append(sf.no_unsafe_claims(text, case.forbidden))
    if case.attack_type == "off_topic":
        safety.append(sf.off_topic_declined(text))
    return (
        _layer("retrieval", retrieval)
        + _layer("deterministic", deterministic)
        + _layer("reference", reference)
        + _layer("safety", safety)
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
    stability_cases: Collection[str] | None = None,
    index: BM25Index | None = None,
    judge: Judge | None = None,
) -> list[CaseRecord]:
    """Run the assistant over `cases`. With `judge`, every run of a judged
    case is graded and gets the judge layer's checks."""
    records = []
    for case in cases:
        runs = []
        for repeat in range(repeats_for(case.id, repeats, stability_cases)):
            answer = assistant.answer(
                client, role, case.question, version, index=index, repeat=repeat, case=case.id
            )
            checks = rag_checks(case, answer)
            graded = None
            if judge is not None and judged(case):
                judgement = judge.grade(
                    case.question,
                    answer.hits,
                    answer.text,
                    case=case.id,
                    version=version,
                    repeat=repeat,
                )
                checks += _layer("judge", judge.checks(judgement))
                graded = JudgeRecord.of(judgement)
            runs.append(
                RunRecord(
                    repeat=repeat,
                    output=answer.text,
                    retrieved=list(answer.retrieved_ids),
                    cited=list(answer.cited_ids),
                    call=CallRecord.from_call(answer.call),
                    checks=checks,
                    judge=graded,
                )
            )
        fields = {"expected", "required_facts", "expected_docs", "forbidden"}
        if case.category == "safety":
            fields |= {"attack_type", "trap_docs"}
        expected = case.model_dump(mode="json", include=fields)
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
    stability_cases: Collection[str] | None = None,
) -> list[CaseRecord]:
    records = []
    for case in cases:
        runs = []
        for repeat in range(repeats_for(case.id, repeats, stability_cases)):
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
    stability_cases: Collection[str] | None = None,
    index: BM25Index | None = None,
    judge: Judge | None = None,
) -> FunctionResults:
    """Run one function with one prompt version over its dataset."""
    if function == "rag":
        records = run_rag(
            client,
            role,
            cases,
            version,
            repeats=repeats,
            stability_cases=stability_cases,
            index=index,
            judge=judge,
        )
    elif function == "triage":
        records = run_triage(
            client, role, cases, version, repeats=repeats, stability_cases=stability_cases
        )
    else:
        raise ValueError(f"unknown function {function!r}; known: {', '.join(EVAL_FUNCTIONS)}")
    prompt = load_prompt(PROMPT_NAMES[function], version)
    graded = judge is not None and function == "rag"
    return FunctionResults(
        function=function,
        version=version,
        mode=str(mode),
        model=role.model,
        prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        dataset=Path(dataset_path).name,
        dataset_sha256=file_sha256(dataset_path),
        repeats=repeats,
        judge_model=judge.role.model if graded else None,
        rubric_sha256=judge.rubric.sha256 if graded else None,
        summary=summarise(function, records),
        cases=records,
    )


def compare_versions(
    judge: Judge,
    cases: Sequence[RagCase],
    first: FunctionResults,
    second: FunctionResults,
    *,
    mode: Mode,
    dataset_path: Path,
    index: BM25Index | None = None,
) -> PairwiseResults:
    """Compare two prompt versions' answers (repeat 0) on every judged case."""
    versions = (first.version, second.version)
    answers = {
        result.version: {record.id: record.runs[0].output for record in result.cases}
        for result in (first, second)
    }
    records = []
    for case in cases:
        if not judged(case):
            continue
        _, hits = assistant.prepare(case.question, versions[0], index=index)
        pair = (answers[versions[0]][case.id], answers[versions[1]][case.id])
        result = judge.compare(case.question, hits, *pair, case=case.id, versions=versions)
        records.append(
            PairwiseCaseRecord.of(case.id, case.category, case.question, pair, result, versions)
        )
    return PairwiseResults(
        function="rag",
        versions=versions,
        mode=str(mode),
        judge_model=judge.role.model,
        rubric_sha256=judge.rubric.sha256,
        dataset=Path(dataset_path).name,
        dataset_sha256=file_sha256(dataset_path),
        summary=summarise_pairwise(records, versions),
        cases=records,
    )


@dataclass(frozen=True)
class RunOutcome:
    pending: bool
    reason: str | None
    results: tuple[FunctionResults, ...]
    written: tuple[Path, ...]
    pairwise: tuple[PairwiseResults, ...] = ()


def run(
    client: ChatModel,
    role: RoleConfig,
    *,
    mode: Mode,
    cassettes_dir: Path | str,
    results_dir: Path | str,
    live_results_dir: Path | str = LIVE_RESULTS_DIR,
    rag_cases: Sequence[RagCase] = (),
    triage_cases: Sequence[TriageCase] = (),
    dataset_paths: Mapping[str, Path],
    versions: Mapping[str, Sequence[str]] | None = None,
    repeats: int = 1,
    stability_cases: Collection[str] | None = None,
    index: BM25Index | None = None,
    judge: RoleConfig | None = None,
    rubric: Rubric | None = None,
) -> RunOutcome:
    """Evaluate every function that has cases, for each chosen prompt version.

    Replay without a manifest returns a pending outcome and calls nothing.
    Results are written only after every function and version has run, so a
    failure (such as a missing recording) leaves `results/` untouched.
    With `judge` (the judge role), RAG runs are graded with `rubric` (default:
    `rubrics/judge.md`) and the RAG prompt versions are compared pairwise.
    """
    mode = Mode(mode)
    if mode is Mode.REPLAY and load_manifest(cassettes_dir) is None:
        return RunOutcome(pending=True, reason=PENDING_RECORDED_RUN, results=(), written=())
    index = index or default_index()
    grader = Judge(client, judge, rubric or load_rubric()) if judge is not None else None
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
            stability_cases=stability_cases,
            index=index,
            judge=grader,
        )
        for function, cases in work
        for version in _versions(versions, function)
    )
    pairwise: tuple[PairwiseResults, ...] = ()
    if grader is not None and rag_cases:
        rag = {result.version: result for result in results if result.function == "rag"}
        pairwise = tuple(
            compare_versions(
                grader,
                rag_cases,
                rag[first],
                rag[second],
                mode=mode,
                dataset_path=dataset_paths["rag"],
                index=index,
            )
            for first, second in pairs_of(_versions(versions, "rag"))
        )
    everything = (*results, *pairwise)
    if mode is Mode.REPLAY:
        written = tuple(write_results(result, results_dir) for result in everything)
    elif mode is Mode.LIVE:
        written = tuple(write_live_results(result, live_results_dir) for result in everything)
    else:
        written = ()
    return RunOutcome(
        pending=False, reason=None, results=results, written=written, pairwise=pairwise
    )
