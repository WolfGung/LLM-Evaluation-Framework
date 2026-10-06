"""Result records, summaries and the results files.

`results/<function>-<version>.json` holds one `FunctionResults`: a record per
case and repeat with every check, and pass rates per layer. With the judge,
each graded run also keeps the judge's verdict, and the summary reports the
judge's reliability. `results/<function>-<v1>-vs-<v2>.json` holds one
`PairwiseResults`: the judge's choice between two prompt versions, case by
case. Only replay results of a recorded run go to `results/`; live results go
to the git-ignored `results-live/`, so every number in `results/` can be
reproduced from the cassettes.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from app.triage import Category, Priority
from llmeval.checks import reference as ref
from llmeval.checks.judge import (
    CRITERIA,
    INCONSISTENCIES,
    OUTCOMES,
    SCORES,
    Judgement,
    OrderJudgement,
    PairwiseResult,
    inconsistency,
    position_bias,
    spearman,
)
from llmeval.checks.retrieval import retrieval_recall_value
from llmeval.client import CallResult
from llmeval.config import JudgeRepeats
from llmeval.perf import Performance, performance
from llmeval.stability import Stability, stability

LAYERS = ("retrieval", "deterministic", "reference", "safety", "judge")
RESULTS_DIR = Path("results")
LIVE_RESULTS_DIR = Path("results-live")
SCHEMA_VERSION = 1

CATEGORY_LABELS = tuple(str(c) for c in Category)
PRIORITY_LABELS = tuple(str(p) for p in Priority)


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CheckRecord(_Record):
    layer: str
    name: str
    passed: bool
    detail: str


class CallRecord(_Record):
    key: str
    model_used: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    cost_usd: float | None
    cost_source: str
    finish_reason: str | None
    empty_reason: str | None
    recorded_at: datetime

    @classmethod
    def from_call(cls, call: CallResult) -> CallRecord:
        return cls(
            key=call.key,
            model_used=call.model_used,
            latency_ms=call.latency_ms,
            prompt_tokens=call.usage.prompt_tokens,
            completion_tokens=call.usage.completion_tokens,
            reasoning_tokens=call.usage.reasoning_tokens,
            cost_usd=call.cost_usd,
            cost_source=call.cost_source,
            finish_reason=call.finish_reason,
            empty_reason=call.empty_reason,
            recorded_at=call.recorded_at,
        )


class JudgeRecord(_Record):
    """The judge's grade of one answer.

    A valid verdict keeps the scores, the judge's own `pass`, the rubric's
    rule applied to the scores (`rule_pass`) and the reasons. An invalid one
    keeps the error, what was wrong and the reply exactly as written.
    """

    scores: dict[str, int] | None
    judge_pass: bool | None
    rule_pass: bool | None
    reasons: str | None
    error: str | None
    detail: str | None
    raw: str | None
    call: CallRecord

    @classmethod
    def of(cls, judgement: Judgement) -> JudgeRecord:
        verdict = judgement.verdict
        return cls(
            scores=verdict.scores if verdict else None,
            judge_pass=verdict.passed if verdict else None,
            rule_pass=judgement.rule_pass,
            reasons=verdict.reasons if verdict else None,
            error=judgement.error,
            detail=None if verdict else judgement.detail,
            raw=None if verdict else judgement.raw,
            call=CallRecord.from_call(judgement.call),
        )


class RunRecord(_Record):
    """One repeat of one case: the output, the call, every check, and the
    judge's grade when the case is graded."""

    repeat: int
    output: str
    error: str | None = None
    retrieved: list[str] | None = None
    cited: list[str] | None = None
    call: CallRecord
    checks: list[CheckRecord]
    judge: JudgeRecord | None = None

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)


class CaseRecord(_Record):
    id: str
    category: str
    input: str
    expected: dict[str, Any]
    runs: list[RunRecord]


class Rate(_Record):
    passed: int
    total: int
    rate: float | None

    @classmethod
    def of(cls, outcomes: Iterable[bool]) -> Rate:
        outcomes = list(outcomes)
        passed = sum(outcomes)
        return cls(
            passed=passed,
            total=len(outcomes),
            rate=round(passed / len(outcomes), 4) if outcomes else None,
        )


class Share(_Record):
    """How many of `total` have a property (not a pass rate)."""

    count: int
    total: int
    rate: float | None

    @classmethod
    def of(cls, flags: Iterable[bool]) -> Share:
        flags = list(flags)
        count = sum(flags)
        return cls(
            count=count, total=len(flags), rate=round(count / len(flags), 4) if flags else None
        )


class LengthCorrelation(_Record):
    """Spearman correlation between answer length (words) and one criterion's score."""

    n: int
    spearman: float | None


class JudgeSummary(_Record):
    """The judge layer of one function and version.

    - `judged`: runs the judge graded;
    - `valid`: valid verdicts among them, the judge's reliability; the rest
      are counted in `invalid_by_kind` (empty, invalid_json, invalid_schema);
    - `rule_pass`: the rubric's pass rule, over valid verdicts only;
    - `pass_disagreements`: valid verdicts whose own `pass` differs from the
      rule on their scores;
    - `empty_answers`: graded runs whose answer was empty. The rubric scores
      them 1 on every criterion, so they stay out of the scores below; they
      still count in `rule_pass`, as failed runs, whatever the judge scored;
    - `mean_scores` and `score_counts`: per criterion, over valid verdicts on
      non-empty answers;
    - `length_score_correlation`: per criterion, between the answer's length
      in words and its score, over valid verdicts on non-empty answers. The
      rubric says length earns nothing, so a strong positive value suggests
      the judge rewards verbosity. It is a cheap check, not proof: longer
      answers can also be more complete;
    - `performance`: latency, tokens and cost of the judge calls (see
      `llmeval.perf`).

    `layers["judge"]` in the summary counts an invalid verdict as a failed
    run, because the answer could not be graded.
    """

    judged: int
    valid: Share
    invalid_by_kind: dict[str, int]
    rule_pass: Rate
    pass_disagreements: int
    empty_answers: int
    mean_scores: dict[str, float | None]
    score_counts: dict[str, dict[str, int]]
    length_score_correlation: dict[str, LengthCorrelation]
    performance: Performance


class Summary(_Record):
    """Pass rates over all runs (every repeat of every case counts once).

    - `layers`: a run passes a layer when every check of that layer passes;
      only runs that have a check in the layer count.
    - `checks`: per check, over the runs it applies to.
    - `all_checks` and `by_category`: runs that pass every check.
    - RAG: `retrieval_recall` is expected documents retrieved / expected
      documents, over cases (the search does not depend on the model).
    - Triage: `accuracy` per label and `confusion` matrices (expected label by
      predicted label, with `invalid` for no usable value).
    - `judge`: the judge layer's reliability and scores, when runs were graded.
    - `stability`: the share of repeated cases whose repeats agree on every
      rule-based check (see `llmeval.stability`); None when no case repeats.
    - `performance`: latency, tokens and cost of the system calls, and their
      tail: the calls at or above p95 next to the fastest other repeat of the
      same case (see `llmeval.perf`); the judge's calls are in
      `judge.performance`.
    """

    cases: int
    runs: int
    all_checks: Rate
    layers: dict[str, Rate]
    checks: dict[str, Rate]
    by_category: dict[str, Rate]
    retrieval_recall: float | None = None
    accuracy: dict[str, float | None] | None = None
    confusion: dict[str, dict[str, dict[str, int]]] | None = None
    judge: JudgeSummary | None = None
    stability: Stability | None = None
    performance: Performance


class FunctionResults(_Record):
    """The content of `results/<function>-<version>.json`.

    `judge_model`, `judge_repeats` (`first`: repeat 0 graded, `all`: every
    repeat) and `rubric_sha256` are set when the runs were graded.
    """

    schema_version: int = SCHEMA_VERSION
    function: str
    version: str
    mode: str
    model: str
    prompt_sha256: str
    dataset: str
    dataset_sha256: str
    repeats: int
    judge_model: str | None = None
    judge_repeats: JudgeRepeats | None = None
    rubric_sha256: str | None = None
    summary: Summary
    cases: list[CaseRecord]


class PairwiseOrderRecord(_Record):
    """One of the two pairwise questions: which version was shown as answer A,
    which letter the judge preferred, and which version that is."""

    shown_as_a: str
    preferred: str | None
    winner: str | None
    reasons: str | None
    error: str | None
    detail: str | None
    raw: str | None
    call: CallRecord

    @classmethod
    def of(cls, order: OrderJudgement, versions: tuple[str, str]) -> PairwiseOrderRecord:
        names = {"v1": versions[0], "v2": versions[1], "tie": "tie"}
        verdict = order.verdict
        return cls(
            shown_as_a=names[order.first],
            preferred=order.preferred,
            winner=names[order.winner] if order.winner else None,
            reasons=verdict.reasons if verdict else None,
            error=order.error,
            detail=None if verdict else order.detail,
            raw=None if verdict else order.raw,
            call=CallRecord.from_call(order.call),
        )


class PairwiseCaseRecord(_Record):
    """One case compared: both answers (repeat 0), the outcome and both orders.

    `outcome` is a version id when both orders preferred it, `tie`,
    `inconsistent` when the orders disagree, `identical` when both versions
    wrote the same text (the judge is not asked, so `orders` is empty), or
    `invalid` when a verdict was invalid. An inconsistent pair is never
    resolved by picking one order.
    """

    id: str
    category: str
    input: str
    answers: dict[str, str]
    outcome: str
    orders: list[PairwiseOrderRecord]

    @classmethod
    def of(
        cls,
        case_id: str,
        category: str,
        question: str,
        answers: tuple[str, str],
        result: PairwiseResult,
        versions: tuple[str, str],
    ) -> PairwiseCaseRecord:
        names = {"v1": versions[0], "v2": versions[1]}
        return cls(
            id=case_id,
            category=category,
            input=question,
            answers=dict(zip(versions, answers, strict=True)),
            outcome=names.get(result.outcome, result.outcome),
            orders=[PairwiseOrderRecord.of(order, versions) for order in result.orders],
        )


class PairwiseSummary(_Record):
    """The comparison of two versions over every compared case.

    - `outcomes`: pairs per outcome (first version, second version, tie,
      inconsistent, identical, invalid);
    - `valid`: pairs whose two verdicts are both valid (the judge's
      reliability), over the pairs the judge was asked;
    - `inconsistent`: inconsistent pairs among the compared ones (valid and
      not identical), and `inconsistent_kinds`: how they disagreed (the same
      position twice, or a tie in one order);
    - `position_bias`: choices of answer A among the choices of a side, over
      compared pairs where both orders chose a side; 0.5 means no lean, 1.0
      means the judge always picks A;
    - `longer_preferred`: over the same choices, how often the judge chose
      the longer answer (pairs of equal length left out). Like the
      length-score correlation, it is a cheap check, not proof: the longer
      answer may also be the better one;
    - `performance`: latency, tokens and cost of the questions asked, with the
      cost per case over every compared case (an identical pair costs
      nothing; see `llmeval.perf`).

    Identical pairs (both versions wrote the same text) are not asked, so
    they have no verdicts and stay out of `valid`, `inconsistent`,
    `position_bias` and `longer_preferred`; they count only in `outcomes`.
    """

    pairs: int
    outcomes: dict[str, int]
    valid: Share
    inconsistent: Share
    inconsistent_kinds: dict[str, int]
    position_bias: Share
    longer_preferred: Share
    performance: Performance


class PairwiseResults(_Record):
    """The content of `results/<function>-<v1>-vs-<v2>.json`."""

    schema_version: int = SCHEMA_VERSION
    function: str
    versions: tuple[str, str]
    mode: str
    judge_model: str
    rubric_sha256: str
    dataset: str
    dataset_sha256: str
    summary: PairwiseSummary
    cases: list[PairwiseCaseRecord]


def summarise(function: str, cases: Sequence[CaseRecord]) -> Summary:
    runs = [(case, run) for case in cases for run in case.runs]
    by_layer: dict[str, list[bool]] = defaultdict(list)
    by_check: dict[str, list[bool]] = {}
    by_category: dict[str, list[bool]] = {}
    for case, run in runs:
        layers: dict[str, bool] = {}
        for check in run.checks:
            layers[check.layer] = layers.get(check.layer, True) and check.passed
            by_check.setdefault(check.name, []).append(check.passed)
        for layer, passed in layers.items():
            by_layer[layer].append(passed)
        by_category.setdefault(case.category, []).append(run.passed)
    ordered_layers = [layer for layer in LAYERS if layer in by_layer] + sorted(
        set(by_layer) - set(LAYERS)
    )
    summary: dict[str, Any] = {
        "cases": len(cases),
        "runs": len(runs),
        "all_checks": Rate.of(run.passed for _, run in runs),
        "layers": {layer: Rate.of(by_layer[layer]) for layer in ordered_layers},
        "checks": {name: Rate.of(outcomes) for name, outcomes in by_check.items()},
        "by_category": {name: Rate.of(outcomes) for name, outcomes in by_category.items()},
    }
    if function == "rag":
        expected = [
            case.expected.get("expected_docs")
            for case in cases
            if case.expected.get("expected_docs")
        ]
        retrieved = [
            case.runs[0].retrieved or [] for case in cases if case.expected.get("expected_docs")
        ]
        total = sum(len(docs) for docs in expected)
        found = sum(
            (retrieval_recall_value(docs, got) or 0) * len(docs)
            for docs, got in zip(expected, retrieved, strict=True)
        )
        summary["retrieval_recall"] = round(found / total, 4) if total else None
    if function == "triage":
        summary["accuracy"] = {
            label: Rate.of(by_check.get(f"{label}_match", [])).rate
            for label in ("category", "priority", "order_id")
        }
        summary["confusion"] = {
            label: ref.confusion_matrix(
                [
                    (case.expected[label], ref.predicted(run.output, label, labels))
                    for case, run in runs
                ],
                labels,
            )
            for label, labels in (("category", CATEGORY_LABELS), ("priority", PRIORITY_LABELS))
        }
    graded = [(run.output, run.judge) for _, run in runs if run.judge is not None]
    if graded:
        graded_cases = len({case.id for case, run in runs if run.judge is not None})
        summary["judge"] = summarise_judge(graded, cases=graded_cases)
    summary["stability"] = stability(function, cases)
    # The repeats of a case send the same request, so the tail can compare them.
    summary["performance"] = performance(
        [run.call for _, run in runs],
        cases=len(cases),
        requests=[(case.id, run.repeat) for case, run in runs],
    )
    return Summary(**summary)


def summarise_judge(
    graded: Sequence[tuple[str, JudgeRecord]], *, cases: int | None = None
) -> JudgeSummary:
    """The judge layer over (answer, judge record) pairs; `cases` is how many
    cases they cover, for the cost per case (None: not given)."""
    records = [record for _, record in graded]
    valid = [record for record in records if record.scores is not None]
    # Valid verdicts on answers with text: the scores and the correlation.
    written = [(answer, record) for answer, record in graded if answer.strip()]
    measured = [
        (len(answer.split()), record.scores)
        for answer, record in written
        if record.scores is not None
    ]
    scored = [scores for _, scores in measured]
    invalid: dict[str, int] = {}
    for record in records:
        if record.error is not None:
            invalid[record.error] = invalid.get(record.error, 0) + 1
    mean_scores = {
        criterion: round(sum(scores[criterion] for scores in scored) / len(scored), 4)
        if scored
        else None
        for criterion in CRITERIA
    }
    score_counts = {
        criterion: {
            str(score): sum(scores[criterion] == score for scores in scored) for score in SCORES
        }
        for criterion in CRITERIA
    }
    correlation = {
        criterion: LengthCorrelation(
            n=len(measured),
            spearman=spearman([w for w, _ in measured], [s[criterion] for _, s in measured]),
        )
        for criterion in CRITERIA
    }
    return JudgeSummary(
        judged=len(records),
        valid=Share.of(record.scores is not None for record in records),
        invalid_by_kind=dict(sorted(invalid.items())),
        rule_pass=Rate.of(bool(r.rule_pass) for r in valid),
        pass_disagreements=sum(r.judge_pass != r.rule_pass for r in valid),
        empty_answers=len(graded) - len(written),
        mean_scores=mean_scores,
        score_counts=score_counts,
        length_score_correlation=correlation,
        performance=performance([record.call for record in records], cases=cases),
    )


def summarise_pairwise(
    cases: Sequence[PairwiseCaseRecord], versions: tuple[str, str]
) -> PairwiseSummary:
    names = {"v1": versions[0], "v2": versions[1]}
    outcomes = {names.get(outcome, outcome): 0 for outcome in OUTCOMES}
    for case in cases:
        outcomes[case.outcome] += 1
    # Identical pairs are not asked; every other pair has two verdicts.
    asked = [case for case in cases if case.outcome != "identical"]
    both_valid = [all(order.error is None for order in case.orders) for case in asked]
    # Pairs the judge really compared: two different answers, two valid verdicts.
    compared = [case for case, valid in zip(asked, both_valid, strict=True) if valid]
    preferences = [(case.orders[0].preferred, case.orders[1].preferred) for case in compared]
    kinds = {kind: 0 for kind in INCONSISTENCIES}
    for first, second in preferences:
        if kind := inconsistency(first, second):
            kinds[kind] += 1
    chose_a, chose = position_bias(preferences)
    return PairwiseSummary(
        pairs=len(cases),
        outcomes=outcomes,
        valid=Share.of(both_valid),
        inconsistent=Share.of(case.outcome == "inconsistent" for case in compared),
        inconsistent_kinds=kinds,
        position_bias=Share(
            count=chose_a, total=chose, rate=round(chose_a / chose, 4) if chose else None
        ),
        longer_preferred=Share.of(_longer_preferred(compared, versions)),
        performance=performance(
            [order.call for case in cases for order in case.orders], cases=len(cases)
        ),
    )


def _longer_preferred(cases: Sequence[PairwiseCaseRecord], versions: tuple[str, str]) -> list[bool]:
    """For each side choice in pairs where both orders chose a side: was it
    the longer answer? Pairs of equal length are left out. A cheap check of
    verbosity bias, not proof: the longer answer may also be the better one."""
    flags = []
    for case in cases:
        if any(order.preferred not in ("A", "B") for order in case.orders):
            continue
        words = {version: len(case.answers[version].split()) for version in versions}
        if words[versions[0]] == words[versions[1]]:
            continue
        longer = max(versions, key=lambda version: words[version])
        flags += [order.winner == longer for order in case.orders]
    return flags


def results_path(results_dir: Path | str, function: str, version: str) -> Path:
    return Path(results_dir) / f"{function}-{version}.json"


def pairwise_path(results_dir: Path | str, function: str, versions: tuple[str, str]) -> Path:
    return Path(results_dir) / f"{function}-{versions[0]}-vs-{versions[1]}.json"


AnyResults = FunctionResults | PairwiseResults


def _write(results: AnyResults, directory: Path | str) -> Path:
    if isinstance(results, PairwiseResults):
        path = pairwise_path(directory, results.function, results.versions)
    else:
        path = results_path(directory, results.function, results.version)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(results.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def write_results(results: AnyResults, results_dir: Path | str) -> Path:
    """Write replay results of a recorded run (the only kind `results/` takes)."""
    if results.mode != "replay":
        raise ValueError(
            f"only replay results go to {results_dir}; got {results.mode} results "
            f"(live results go to {LIVE_RESULTS_DIR}/, a recording is replayed first)"
        )
    return _write(results, results_dir)


def write_live_results(results: AnyResults, live_dir: Path | str) -> Path:
    """Write results of a live run to the git-ignored live directory."""
    if results.mode != "live":
        raise ValueError(f"only live results go to {live_dir}; got {results.mode} results")
    return _write(results, live_dir)
